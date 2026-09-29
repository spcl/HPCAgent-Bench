! Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
! SPDX-License-Identifier: GPL-3.0-or-later
!
! The Fortran twin of openmp_probe.c, for containers/lib/one_openmp_gate.py: built with `gfortran -fopenmp`
! (and `flang -fopenmp` where present), same entry point, same check bits and result slots.
module openmp_probe_state
    implicit none
    integer :: thread_private
    !$omp threadprivate(thread_private)
end module openmp_probe_state

function omp_probe(nthreads, result) bind(c, name="omp_probe") result(failed)
    use, intrinsic :: iso_c_binding, only: c_int, c_long
    use omp_lib
    use openmp_probe_state
    implicit none
    integer(c_int), value :: nthreads
    integer(c_int), intent(inout) :: result(*)
    integer(c_int) :: failed

    integer, parameter :: n = 4096, max_threads = 256
    integer, parameter :: static_team = 1, dynamic_team = 2, guided_team = 3, task_team = 4, barrier_threads = 5
    integer, parameter :: check_static = 0, check_dynamic = 1, check_guided = 2, check_runtime = 3, &
        check_collapse = 4, check_ordered = 5, check_sections = 6, check_reduction_minmax = 7, check_tasks = 8, &
        check_taskloop = 9, check_task_depend = 10, check_atomic = 11, check_critical = 12, check_simd = 13, &
        check_parallel_do_simd = 14, check_lock = 15, check_nest_lock = 16, check_threadprivate = 17, &
        check_max_threads = 18
    integer(c_long) :: total, counter, guarded, done, looped, chained
    integer :: i, j, team, cursor, order(64), first, second
    integer :: seen(0:max_threads - 1)
    logical :: in_order, chain_ok, intact
    real(8) :: values(n), lowest, highest, vsum
    integer(omp_lock_kind) :: lock
    integer(omp_nest_lock_kind) :: nest

    failed = 0
    result(1:5) = 0
    values = 1.0d0
    call omp_set_num_threads(nthreads)

    total = 0; team = 0
    !$omp parallel do schedule(static) reduction(+:total) reduction(max:team)
    do i = 0, n - 1
        total = total + i
        team = omp_get_num_threads()
    end do
    !$omp end parallel do
    result(static_team) = team
    if (total /= int(n, c_long) * (n - 1) / 2) failed = ibset(failed, check_static)

    total = 0; team = 0
    !$omp parallel do schedule(dynamic, 16) reduction(+:total) reduction(max:team)
    do i = 0, n - 1
        total = total + i
        team = omp_get_num_threads()
    end do
    !$omp end parallel do
    result(dynamic_team) = team
    if (total /= int(n, c_long) * (n - 1) / 2) failed = ibset(failed, check_dynamic)

    total = 0; team = 0
    !$omp parallel do schedule(guided) reduction(+:total) reduction(max:team)
    do i = 0, n - 1
        total = total + i
        team = omp_get_num_threads()
    end do
    !$omp end parallel do
    result(guided_team) = team
    if (total /= int(n, c_long) * (n - 1) / 2) failed = ibset(failed, check_guided)

    total = 0
    !$omp parallel do schedule(runtime) reduction(+:total)
    do i = 0, n - 1
        total = total + i
    end do
    !$omp end parallel do
    if (total /= int(n, c_long) * (n - 1) / 2) failed = ibset(failed, check_runtime)

    total = 0
    !$omp parallel do collapse(2) reduction(+:total)
    do i = 0, 63
        do j = 0, 63
            total = total + int(i, c_long) * j
        end do
    end do
    !$omp end parallel do
    if (total /= (64_c_long * 63 / 2) * (64_c_long * 63 / 2)) failed = ibset(failed, check_collapse)

    cursor = 0
    !$omp parallel do ordered schedule(dynamic)
    do i = 0, 63
        !$omp ordered
        cursor = cursor + 1
        order(cursor) = i
        !$omp end ordered
    end do
    !$omp end parallel do
    in_order = .true.
    do i = 0, 63
        in_order = in_order .and. order(i + 1) == i
    end do
    if (.not. in_order) failed = ibset(failed, check_ordered)

    first = 0; second = 0
    !$omp parallel sections
    !$omp section
    first = 1
    !$omp section
    second = 1
    !$omp end parallel sections
    if (first /= 1 .or. second /= 1) failed = ibset(failed, check_sections)

    lowest = 1d300; highest = -1d300
    !$omp parallel do reduction(min:lowest) reduction(max:highest)
    do i = 0, n - 1
        lowest = min(lowest, real(i, 8))
        highest = max(highest, real(i, 8))
    end do
    !$omp end parallel do
    if (lowest /= 0d0 .or. highest /= real(n - 1, 8)) failed = ibset(failed, check_reduction_minmax)

    done = 0; looped = 0; chained = 0; chain_ok = .true.; team = 0
    !$omp parallel
    !$omp single
    team = omp_get_num_threads()
    do i = 1, 64
        !$omp task shared(done)
        !$omp atomic
        done = done + 1
        !$omp end task
    end do
    !$omp taskwait
    if (done /= 64) failed = ibset(failed, check_tasks)

    !$omp taskloop grainsize(8) shared(looped)
    do i = 1, 256
        !$omp atomic
        looped = looped + 1
    end do
    !$omp end taskloop
    if (looped /= 256) failed = ibset(failed, check_taskloop)

    do i = 0, 15
        !$omp task depend(inout: chained) shared(chained, chain_ok) firstprivate(i)
        chain_ok = chain_ok .and. chained == i
        chained = i + 1
        !$omp end task
    end do
    !$omp taskwait
    if ((.not. chain_ok) .or. chained /= 16) failed = ibset(failed, check_task_depend)
    !$omp end single
    !$omp end parallel
    result(task_team) = team

    counter = 0
    !$omp parallel do
    do i = 1, n
        !$omp atomic
        counter = counter + 2
    end do
    !$omp end parallel do
    if (counter /= 2_c_long * n) failed = ibset(failed, check_atomic)

    guarded = 0
    !$omp parallel do
    do i = 1, n
        !$omp critical (probe_guard)
        guarded = guarded + 1
        !$omp end critical (probe_guard)
    end do
    !$omp end parallel do
    if (guarded /= n) failed = ibset(failed, check_critical)

    vsum = 0d0
    !$omp simd reduction(+:vsum)
    do i = 1, n
        vsum = vsum + values(i)
    end do
    if (vsum /= real(n, 8)) failed = ibset(failed, check_simd)
    vsum = 0d0
    !$omp parallel do simd reduction(+:vsum)
    do i = 1, n
        vsum = vsum + values(i)
    end do
    !$omp end parallel do simd
    if (vsum /= real(n, 8)) failed = ibset(failed, check_parallel_do_simd)

    counter = 0
    call omp_init_lock(lock)
    !$omp parallel do
    do i = 1, n
        call omp_set_lock(lock)
        counter = counter + 1
        call omp_unset_lock(lock)
    end do
    !$omp end parallel do
    call omp_destroy_lock(lock)
    if (counter /= n) failed = ibset(failed, check_lock)

    counter = 0
    call omp_init_nest_lock(nest)
    !$omp parallel do
    do i = 1, n
        call omp_set_nest_lock(nest)
        call omp_set_nest_lock(nest)
        counter = counter + 1
        call omp_unset_nest_lock(nest)
        call omp_unset_nest_lock(nest)
    end do
    !$omp end parallel do
    call omp_destroy_nest_lock(nest)
    if (counter /= n) failed = ibset(failed, check_nest_lock)

    intact = .true.
    !$omp parallel reduction(.and.:intact)
    thread_private = omp_get_thread_num() + 1
    !$omp barrier
    intact = thread_private == omp_get_thread_num() + 1
    !$omp end parallel
    if (.not. intact) failed = ibset(failed, check_threadprivate)

    seen = 0
    !$omp parallel
    if (omp_get_thread_num() < max_threads) seen(omp_get_thread_num()) = 1
    !$omp barrier
    !$omp end parallel
    result(barrier_threads) = sum(seen)

    if (omp_get_max_threads() /= nthreads) failed = ibset(failed, check_max_threads)
end function omp_probe
