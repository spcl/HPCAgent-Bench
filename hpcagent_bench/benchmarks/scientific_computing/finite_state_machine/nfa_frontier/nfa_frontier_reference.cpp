/*
 * Adapted from VASim -- the Virtual Automata Simulator (github.com/jackwadden/VASim),
 * MIT license (Jack Wadden, University of Virginia), the reference engine of the ANMLZoo
 * automata-processing benchmark suite (github.com/jackwadden/ANMLZoo; Wadden, Dang,
 * Brunelle et al., IISWC'16). Verbatim from src/automata.cpp and src/element.cpp at
 * commit 3a9781bf33909166825b27506da339f985140416: `Automata::simulate` and the three steps it runs
 * per input symbol, plus the successor walk step 2 calls. Not the scoring oracle -- the
 * numpy reference remains the correctness oracle.
 */


// ---- src/automata.cpp ----

/**
 * Simulates the automata on input string. Starts at start_index and runs for length symbols.
 */
void Automata::simulate(uint8_t *inputs, uint64_t start_index, uint64_t length, uint64_t total_length) {

    cycle = start_index;

    // primes all data structures for simulation
    initializeSimulation();
    
    // for all inputs
    for(uint64_t i = start_index; i < start_index + length; i = i + 1) {

        // set end of data flag if its the last byte
        if( i == total_length - 1 ) {
            setEndOfData(true);
        }    
        // set end of data flag if the byte is a "\n"
        else if( inputs[i] == (uint32_t)'\n' ) {
            setEndOfData(true);
        }
        // unset end of data otherwise
        else {
            setEndOfData(false);
        }

        // measure progress on longer runs
        if(!quiet) {

            if(i % 10000 == 0) {
                if(i != 0) {
                    cout << "\x1B[2K"; // Erase the entire current line.
                    cout << "\x1B[0E";  // Move to the beginning of the current line.
                }

                cout << "  Progress: " << i << " / " << length << "\r";
                flush(cout);
                //
            }
        }
        simulate(inputs[i]);
    }

    if(!quiet) {
        cout << "\x1B[2K"; // Erase the entire current line.
        cout << "\x1B[0E";  // Move to the beginning of the current line.
        cout << "  Progress: " << length << " / " << length << "\r";
        flush(cout);
        cout << endl;
    }
 
    if(profile) {

        cout << endl << "Dynamic Statistics: " << endl;

        // cal average active set
        uint64_t sum = 0;
        for(uint32_t acts : activatedHist){
            sum += (uint64_t)acts;
        }

        cout << "  Average Active Set: " << (double)sum / (double)length << endl;
        for(uint32_t acts : activatedHist){
            sum += (uint64_t)acts;
        }

        // cal distribution

        // build histogram of activations
        buildActivationHistogram("activation_hist.out");        
        
        // print activation stats
        calcEnableDistribution();
        
        // write to file
        writeIntVectorToFile(enabledHist, "enabled_per_cycle.out");
        writeIntVectorToFile(activatedHist, "activated_per_cycle.out");
    
        cout << endl;
    }
}

/**
 * Simulates the automata on a single input symbol.
 */
void Automata::simulate(uint8_t symbol) {

    
    // -----------------------------
    // Step 1: if STEs are enabled and we match, activate
    computeSTEMatches(symbol);
    // -----------------------------

    
    // Activation Statistics
    if(profile){
        profileActivations();
    }

    // Debug state
    if(dump_state && (dump_state_cycle == cycle)){
        dumpSTEState("stes_" + to_string(cycle) + ".state");
    }

    // -----------------------------
    // Step 2: enable children of matching STEs
    enableSTEMatchingChildren();
    // -----------------------------


    // -----------------------------
    // Step 3:  enable all-input start states
    enableStartStates(end_of_data);
    // -----------------------------

    
    // -----------------------------
    // Step 4: special element computation
    if(specialElements.size() > 0){        
        specialElementSimulation2();

        if(dump_state && (dump_state_cycle == cycle)){
            dumpSpecelState("specels_" + to_string(cycle) + ".state");
        }
    }
    // -----------------------------

    // Enabled Statistics
    if(profile){
        profileEnables();
    }
    
    // advance cycle count
    tick();
}

/**
 * If an STE is enabled and matches on the current input, activate. If the STE is a report STE, record a report in the report vector. 
 */
void Automata::computeSTEMatches(uint8_t symbol) {

    //for each enabled ste
    while(!enabledSTEs.empty()) {

        STE * s = static_cast<STE *>(enabledSTEs.back());

        // if we match on the input character
        // the STE will activate and we record this
        // ste should also report
        if(s->match(symbol)) {

            //activate and push to queue only if we werent already
            if(!s->isActivated()) {
                s->activate();
                activatedSTEs.push_back(s);
            }

            if(profile)
                activationVector[cycle].push_back(s->getId());

            // report
            if(report && s->isReporting()) {
                if(s->isEod()) {
                    if(end_of_data)
                        reportVector.push_back(make_pair(cycle, s->getId()));
                }else{
                    reportVector.push_back(make_pair(cycle, s->getId()));
                }
            }

        }

        //disable 
        s->disable();

        // remove STE from the queue
        enabledSTEs.pop_back();        
    }
}

/**
 * Propagate activation signal of STEs that match on the current input symbol. Enables Element children of active STEs.
 */
void Automata::enableSTEMatchingChildren() {

    //for each activated ste
    while(!activatedSTEs.empty()) {

        STE *s = activatedSTEs.back();
        // remove from activated queue
        activatedSTEs.pop_back();

        s->enableChildSTEs(&enabledSTEs);

        if(specialElements.size() > 0)
            s->enableChildSpecialElements(&enabledSpecialElements);

        // suggest that the STE deactivate
        // if we don't, add to the queue
        if(!s->deactivate()) {
            // don't mark for removal from activated map
            latchedSTEs.push_back(s);
        }
        
    }
    
    // refil activated elements
    while(!latchedSTEs.empty()) {
        activatedSTEs.push_back(latchedSTEs.back());
        latchedSTEs.pop_back();
    }
}

/**
 * Enable all elements that are start states. Start states initiate computation by being enabled on the first cycle (for start-of-data type) or every cycle (for all-input type).
 */
void Automata::enableStartStates(bool enableStartOfData) {

    //for each start element
    for(STE * s: starts) {

        // Enable if start is "all input"
        if(s->startIsAllInput() || (enableStartOfData && s->startIsStartOfData())) { 
           
            // add to enabled queue if we were not already enabled
            if(!s->isEnabled()){
                s->enable();
                enabledSTEs.push_back(static_cast<Element *>(s));
            }
        }
    }

}

// ---- src/element.cpp ----

void Element::enableChildSTEs(Stack<Element*> *enabledSTEs) {
    
    for(const pair<Element *, string> e : outputSTEPointers) {

        Element * child = e.first;
        
        // only enable if not previously enabled
        if(!child->isEnabled()){
            enabledSTEs->push_back(child);
            static_cast<STE *>(child)->enable();        
        }
    }
}
