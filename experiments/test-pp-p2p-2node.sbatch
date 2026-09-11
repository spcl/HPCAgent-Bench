#!/usr/bin/env bash
#SBATCH --job-name=pp-p2p-2node
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --output=/capstor/scratch/cscs/ybudanaz/x86_64/rccl-ofi/logs/pp-p2p-%j.out
#SBATCH --error=/capstor/scratch/cscs/ybudanaz/x86_64/rccl-ofi/logs/pp-p2p-%j.err
#
# Bisect the CXI env against the failure that kills kimi pipeline parallelism.
#
# Inter-node COLLECTIVES were already proved to ride CXI (job 590212: cxi
# provider, AWS Libfabric, 85.1 GB/s bus at 2n x 4r). Job 590270 then showed that is not enough:
# vLLM's PP handoff -- torch.distributed.irecv on a size-4 group, which makes RCCL build a fresh
# 2-rank communicator -- died with `ncclInternalError: Internal check failed` (RCCL 2.30.4,
# aws-ofi-nccl 1.20.0), taking the engine with it. Collectives healthy, P2P dead.
#
# So the question is which of the CXI/RCCL knobs added in b37cb495 the 2-rank P2P path cannot take.
# Each variant below runs the same probe under a different env, in ONE allocation: about two minutes
# each, against three hours for a vLLM probe that answers the same question.
#
#   sbatch test-pp-p2p-2node.sbatch
#
# Each variant prints `allreduce=<v> p2p_size2=<v> p2p_size4=<v>`, and the two P2P columns are the
# point. torch mints a lazy 2-rank communicator only when the group is WIDER than the pair actually
# exchanging data, so:
#
#   size2 PASS + size4 FAIL -> only the lazy sub-communicator is broken. Pipeline parallelism works
#                              at pp=2 (PP group size 2), which is what probe 589497 already showed
#                              passing where pp=4 failed. Fix is the topology, not the env.
#   both FAIL               -> inter-node P2P is broken outright; pp is unavailable at any width.
#   both PASS               -> the env under test is clean; look at the vLLM side instead.
#
# A variant with size4 PASS names a knob to drop from the EDF. If v0-current passes outright, the
# bug is NOT in the env -- do not go on to re-tune the EDF in that case.
#
# NOTE on an earlier version of this gate: the probe's chain used to WRAP AROUND, and every rank
# issued its irecv before its isend. Creating the lazy 2-rank communicator is itself blocking and
# collective over that pair, so the ring deadlocked on communicator creation and every variant
# reported a TCPStore rendezvous timeout -- a bug in the probe, not in RCCL, and nothing about the
# env could be concluded from it. The chain is a line now, which is also what vLLM's pipeline is.

set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and dumps land in the CWD.
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
RUN_DIR="${SCRATCH}/rccl-ofi/logs/${SLURM_JOB_ID}"
# NOT dirname "${BASH_SOURCE[0]}": slurmd runs a spooled copy out of /var/spool, so that resolves to
# a directory holding nothing but the script itself. SLURM_SUBMIT_DIR is where sbatch was invoked.
PROBE_PY="${PROBE_PY:-${SLURM_SUBMIT_DIR:-.}/pp-p2p-probe.py}"
INFERENCE_EDF="${INFERENCE_EDF:-vllm-latest}"

MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
RANKS_PER_NODE="${RANKS_PER_NODE:-4}"
# Room for the sustained stage (SUSTAIN_SECONDS) on top of init, group creation and the one-shot
# stages. Too tight a wall clock reports a FAIL that is only the probe being cut off mid-run.
STAGE_TIMEOUT="${STAGE_TIMEOUT:-600}"
# Drive mixed TP/PP traffic for this long. Kimi served for about 2.5 minutes before it stopped, so
# a shorter window than that cannot claim the fabric survives the load.
SUSTAIN_SECONDS="${SUSTAIN_SECONDS:-180}"
# Repetitions per (variant, fill) cell. The failure is INTERMITTENT -- see the CELL_PASS comment
# below -- so a single run per cell is not a measurement. Keep this >= 3 for any claim about a fix.
REPEATS="${REPEATS:-5}"
# Fraction of free GPU memory the probe holds before building any communicator. 0 = empty-GPU
# control. The interesting run is a high value: vLLM creates its pipeline communicator with weights
# and KV cache already resident, and registration is what failed there (cxil_map), not transport.
# Two levels, and the pair is the whole experiment. 0.70 is what kimi now serves at; 0 is the same
# traffic with the memory pressure removed. Both FAIL -> the cause is the concurrent traffic itself
# and no utilization number fixes it. Only 0.70 FAILs -> it is memory pressure and the lever is
# --gpu-memory-utilization. Both PASS -> the fabric is not the cause and the next suspect is vLLM.
FILL_FRACTION="${FILL_FRACTION:-0 0.70}"

mkdir -p "${RUN_DIR}"
[[ -f "${PROBE_PY}" ]] || { echo "FAIL: missing ${PROBE_PY}" >&2; exit 2; }

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if (( ${#NODES[@]} != 2 )); then
    echo "FAIL: need exactly 2 nodes, allocation has ${#NODES[@]}" >&2
    exit 2
fi

MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --nodelist="${NODES[0]}" \
    bash -c "ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1")
[[ -n "${MASTER_ADDR}" ]] || { echo "FAIL: no hsn0 address on ${NODES[0]}" >&2; exit 2; }

cat <<EOF
job:      ${SLURM_JOB_ID}
nodes:    ${NODES[*]}
edf:      ${INFERENCE_EDF}
probe:    ${PROBE_PY}
master:   ${MASTER_ADDR}:${MASTER_PORT}
run_dir:  ${RUN_DIR}
EOF

# Variants, cheapest hypothesis first. Each is shell run INSIDE the container before the probe, so
# it edits the env the EDF established. Names are what the summary prints.
#
#   v0-current   reproduce with the EDF exactly as committed -- if this passes, stop, the env is innocent
#   v1-nochan    NCCL_NCHANNELS_PER_NET_PEER=4 asks for 4 channels to every net peer; a 2-rank comm
#                created after a 16-rank one is where that runs out of CXI resources first
#   v2-nocrossnic  NCCL_CROSS_NIC=1 lets a route leave on one NIC and arrive on another
#   v3-nordzv    the four FI_CXI_RDZV_* settings force rendezvous for every message size; a 2-rank
#                P2P comm is the one place that has no collective algorithm to hide a bad choice
#   v4-minimal   drop all CXI/MR tuning, keep only what SELECTS the fabric
#   v5-min-hybrid  minimal plus the one match-mode setting that is a correctness fix, not a tuning knob
# VARIANTS overrides the sweep, e.g. VARIANTS=v0-current to run the committed env alone.
# Default is the committed env alone. The env bisect is DONE -- job 590369 cleared every knob at
# fill <= 0.70 -- so re-running six variants only dilutes the question now on the table, which is
# whether sustained mixed traffic survives at all. Set VARIANTS to bisect the env again if a
# sustained failure turns out to be env-sensitive.
read -r -a VARIANT_NAMES <<<"${VARIANTS:-v0-current}"

TUNING_VARS="FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_DISABLE_HOST_REGISTER \
FI_CXI_RX_MATCH_MODE FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD FI_CXI_RDZV_PROTO FI_CXI_RDZV_EAGER_SIZE \
FI_CXI_RDZV_GET_MIN FI_CXI_RDZV_THRESHOLD FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE \
FI_MR_CACHE_MONITOR NCCL_CROSS_NIC NCCL_NCHANNELS_PER_NET_PEER"

variant_env() {
    case "$1" in
        v0-current)    echo ":" ;;
        v1-nochan)     echo "unset NCCL_NCHANNELS_PER_NET_PEER" ;;
        v2-nocrossnic) echo "unset NCCL_CROSS_NIC" ;;
        v3-nordzv)     echo "unset FI_CXI_RDZV_PROTO FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_GET_MIN FI_CXI_RDZV_THRESHOLD" ;;
        v4-minimal)    echo "unset ${TUNING_VARS}" ;;
        v5-min-hybrid) echo "unset ${TUNING_VARS}; export FI_CXI_RX_MATCH_MODE=hybrid" ;;
        # The committed env plus the one variable RCCL itself asks for. At fill 0.70 job 590674
        # failed the very first collective with "RCCL_USE_AMD_SMI_LIB not set, but HIP_FABRIC_API is
        # defined ... Rerun with RCCL_USE_AMD_SMI_LIB=1 to enable AMD SMI and UALoE fabric support",
        # and the identical line is in every dead kimi arm. A big allocation pushes RCCL onto the
        # fabric-backed path, which it can only drive through AMD SMI -- hence empty GPU fine,
        # loaded GPU dead, with no fabric error at all in between.
        v6-amdsmi)     echo "export RCCL_USE_AMD_SMI_LIB=1" ;;
        # Both variables the AMD guidance pairs: AMD SMI for topology/telemetry, and the scratch
        # reclaim opt-out for firmware that fails to return RCCL scratch under tight VRAM.
        v7-amdsmi-noscratch) echo "export RCCL_USE_AMD_SMI_LIB=1 HSA_NO_SCRATCH_RECLAIM=1" ;;
        v8-noscratch)  echo "export HSA_NO_SCRATCH_RECLAIM=1" ;;
        # --- memory-registration cache, the best fit for a SILENT wrong answer ---
        # The EDF caches registrations essentially without bound (MAX_COUNT=524288, MAX_SIZE=-1) and
        # invalidates them with FI_MR_CACHE_MONITOR=userfaultfd. The cache is keyed by VIRTUAL
        # address, so if the monitor ever misses an unmap, a later allocation that reuses the address
        # inherits a registration pointing at the wrong physical pages: the RDMA still "succeeds" and
        # the data is wrong. That predicts everything the fill threshold predicted -- a big ballast
        # allocation churns the address space, making reuse likely -- and additionally predicts the
        # INTERMITTENCY, which a occupancy threshold does not.
        v9-mrcache-off)  echo "export FI_MR_CACHE_MAX_COUNT=0" ;;
        v10-mr-memhooks) echo "export FI_MR_CACHE_MONITOR=memhooks" ;;
        v11-mr-disabled) echo "export FI_MR_CACHE_MONITOR=disabled" ;;
        # LL/LL128 signal data-readiness with in-band flags in a shared buffer; a missed flag is read
        # as valid stale data rather than an error. Simple uses explicit sync and cannot do that.
        v12-proto-simple) echo "export NCCL_PROTO=Simple" ;;
        # The documented one. NVIDIA/nccl#2001: silent cross-node allreduce corruption present in
        # 2.27.3+ and 2.29.2, absent in 2.26.6, on HPE Cray EX + Slingshot 11 through aws-ofi-nccl,
        # about 30% of 1 MiB messages. LL128 has been the default since 2.27 and neither EDF sets
        # NCCL_PROTO, so every run so far had it live. Our RCCL is 2.27.7 (campaign image) and
        # 2.30.4 (this one) -- both inside the affected band. CSCS and Sigma2 both publish
        # NCCL_PROTO=^LL128 as the workaround; it keeps Simple and LL, so it costs less bandwidth
        # than v12. Run v0-current beside it: the claim is a RATE difference, not a verdict.
        v14-noll128)     echo "export NCCL_PROTO=^LL128" ;;
        # GPU Direct RDMA off, staged through host memory. 590739's own RCCL logs show the EDF's
        # "we never asked for GDR" assumption is false: omitting NCCL_NET_GDR_LEVEL lets RCCL
        # auto-negotiate it, and it chose GDRDMA for 709 of 717 channels on a 6.4 kernel. The
        # comm reached "Init COMPLETE" and only then stopped -- bootstrap goes over host memory,
        # payload goes over GDRDMA -- which is why a 2-element broadcast hangs as readily as a
        # 3584-element recv. Path-dependent, not size-dependent: no NCCL_PROTO value explains that.
        v15-nogdr)       echo "export NCCL_NET_GDR_LEVEL=0" ;;
        # The full combination the kimi arms now run, so the probe measures what is deployed.
        v16-simple-nogdr) echo "export NCCL_PROTO=Simple; export NCCL_NET_GDR_LEVEL=0" ;;
        # MI300A is an APU: host and device share physical memory and XNACK lets the GPU fault pages
        # in and out. A page that migrates while its region is registered for RDMA corrupts silently.
        v13-noxnack)     echo "export HSA_XNACK=0" ;;
        *)             echo "echo 'FAIL: unknown variant $1' >&2; exit 2" ;;
    esac
}

run_variant() {
    local name="$1" fill="$2" rep="$3"
    local label="${1}@fill${2}#${3}" log="${RUN_DIR}/${1}-fill${2}-r${3}.log"

    printf '\n===== variant %s (fill %s) rep %s =====\n' "${name}" "${fill}" "${rep}"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
        --gpus-per-node=4 --cpus-per-task=32 \
        --environment="${INFERENCE_EDF}" \
        bash -c "
            $(variant_env "${name}")
            timeout --signal=TERM --kill-after=15s ${STAGE_TIMEOUT}s \
            torchrun --nnodes=2 --nproc-per-node=${RANKS_PER_NODE} \
                --node-rank=\${SLURM_NODEID} \
                --master-addr='${MASTER_ADDR}' --master-port='${MASTER_PORT}' \
                '${PROBE_PY}' --timeout-seconds 90 --fill-fraction ${fill} \
                --sustain-seconds ${SUSTAIN_SECONDS}
        " 2>&1 | tee "${log}" || true

    verdict "${label}" "${log}" "${name}@fill${fill}"
}

# Report the two stages separately. A single pass/fail line would hide the one distinction that
# matters here: allreduce PASS + p2p FAIL means the fabric is fine and only the 2-rank path is broken.
verdict() {
    local name="$1" log="$2" cell="$3"
    local allreduce p2p pair provider

    # PASS needs rank 0's PASS *and* no rank reporting FAIL: a remote rank can get a silently wrong
    # answer while rank 0 gets the right one, which is exactly what memory pressure produced.
    stage_verdict() {
        grep -q "^stage_$1=FAIL" "${log}" && { echo FAIL; return; }
        grep -q "^stage_$1=PASS" "${log}" && echo PASS || echo FAIL
    }
    allreduce=$(stage_verdict allreduce)
    pair=$(stage_verdict p2p_size2)
    p2p=$(stage_verdict p2p_size4)
    sustained=$(stage_verdict sustained)
    # `|| true` for the same reason as the failure dump below: under `set -o pipefail` this fails
    # when grep matches NOTHING, which is exactly what an earlier stage failing produces -- so the
    # run died at the first failing variant, taking every later variant with it (jobs 590674, 590679).
    rounds=$(grep -oE '^sustained_rounds_total=[0-9]+' "${log}" | tail -1 | cut -d= -f2 || true)
    grep -q 'Using network Socket'  "${log}" && provider=SOCKET || provider=cxi
    grep -q 'Selected provider is cxi' "${log}" || provider=UNKNOWN

    printf -- '-- %s: allreduce=%s p2p_size2=%s p2p_size4=%s sustained=%s rounds=%s net=%s\n' \
        "${name}" "${allreduce}" "${pair}" "${p2p}" "${sustained}" "${rounds:-0}" "${provider}"
    # `|| true`: under `set -o pipefail` this pipeline fails when the grep matches NOTHING, which is
    # precisely the passing case -- it ended the first corrected run after one variant.
    grep -E '^(failed_stage|failure)=' "${log}" | sed 's/^/   | /' | head -4 || true

    SUMMARY+=("${name}: allreduce=${allreduce} p2p_size2=${pair} p2p_size4=${p2p} \
sustained=${sustained} rounds=${rounds:-0} net=${provider}")
    # The sustained stage is what decides now. The one-shot stages already pass at fill 0.70 (job
    # 590369) and kimi still died -- after serving, not at init -- so "communicator builds" is no
    # longer evidence of anything.
    CELL_TOTAL["${cell}"]=$(( ${CELL_TOTAL["${cell}"]:-0} + 1 ))
    if [[ "${sustained}" == PASS && "${provider}" == cxi ]]; then
        CELL_PASS["${cell}"]=$(( ${CELL_PASS["${cell}"]:-0} + 1 ))
        PASSING+=("${name}")
    fi
}

SUMMARY=()
PASSING=()
# Pass RATE per (variant, fill) cell, not a single verdict. The same cell -- v0-current at fill 0.70
# -- came back FAIL (hard ncclInternalError, job 590674), FAIL (silent wrong answer 10.0, job 590679)
# and PASS (154852 sustained rounds, job 590681). One sample per cell measures nothing when the
# failure is intermittent, and it is exactly how a variant gets crowned as "the fix" by luck.
declare -A CELL_PASS CELL_TOTAL
read -r -a FILL_LEVELS <<<"${FILL_FRACTION}"
for fill in "${FILL_LEVELS[@]}"; do
    for name in "${VARIANT_NAMES[@]}"; do
        for rep in $(seq 1 "${REPEATS}"); do
            run_variant "${name}" "${fill}" "${rep}"
        done
    done
done

printf '\n===== summary =====\n'
printf '%s\n' "${SUMMARY[@]}"

printf '\n===== pass rate per cell (%s reps each) =====\n' "${REPEATS}"
clean_cells=()
for cell in $(printf '%s\n' "${!CELL_TOTAL[@]}" | sort); do
    pass="${CELL_PASS[${cell}]:-0}" total="${CELL_TOTAL[${cell}]}"
    printf -- '-- %-34s %s/%s\n' "${cell}" "${pass}" "${total}"
    # A fix has to survive EVERY repetition. Anything less is the intermittency this sweep exists
    # to measure, and calling it a fix is how a lucky sample becomes a deployment decision.
    (( pass == total )) && clean_cells+=("${cell}")
done

if (( ${#clean_cells[@]} == 0 )); then
    echo "PP_P2P_GATE: FAIL -- no cell passed every repetition."
    echo "Logs: ${RUN_DIR}"
    exit 1
fi
echo "PP_P2P_GATE: PASS -- clean across all reps: ${clean_cells[*]}"
echo "Logs: ${RUN_DIR}"
exit 0
