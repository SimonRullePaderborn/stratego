"""Text of version 2 of the design page, sections 7 to 15. Filled into the @@NAME@@ slots."""
from design_v2_part1 import M, U, E, D, Q, table, h2


def blocks(b):
    pre = b.pre
    out = {}

    # ------------------------------------------------------------------ 7
    out["SEARCH"] = h2(7, "search", "Search") + f'''
<p>The rule can only learn what search has found, so search is where compute goes. A trunk is one training episode. At every cut, many clones of the team are started from the same saved state, each with its own strategy and world, and three stages decide what is kept.</p>
''' + pre([
        "round",
        ("    draw 32 trunks from the training mixture", "map, team, start cells, batteries"),
        "    for each trunk, until 96 steps have passed",
        ("        cut: save simulator and lattice just before a target is drawn", "at reset, after a capture or a relocation"),
        ("        screen     M strategies in W1 worlds, closed loop", "the same worlds for all"),
        ("                   V[σ] ← mean over worlds of γ^(steps to capture)", "0 without a capture"),
        ("        confirm    the best E strategies and the greedy one in W2 fresh worlds", ""),
        ("                   σ* ← the best of them, if it passes the test below", "else store nothing"),
        ("        harvest    σ* in W3 further fresh worlds", "only these give weights and credit"),
        ("        continue the trunk from the end state of one clone", "see below"),
        "    learn from the harvested legs (§9)",
    ], 79) + f'''
<figure>
  <div class="scroll diagram">@@FIGLEG@@</div>
  <figcaption>One leg with a hidden target. Left, confirmation: the strategies that survived screening are run from the same snapshot in the same fresh worlds, and one of them beats the greedy rule. Right, credit in the simulator: the kept leg is replayed with recorded actions restored one drone at a time in a random order while the remaining drones act at random; each drone's share is what restoring it adds (§8). An observer's actions add nothing here, since the others repeat what they did anyway. Its share comes from the second test of §8.</figcaption>
</figure>
<ul>
  <li><strong>A cut is taken only at a moment when nothing hidden has been decided yet,</strong> so that every world is equally possible from the team's point of view. No belief filter is written by hand, and no comparison ever uses the hidden position. A second cut at the first sighting, where only the target's tie-breaks remain unknown, is a dial {U}.</li>
  <li><strong>A clone runs</strong> until it captures, until 48 steps have passed (the leg limit H), or until the trunk's 96th step.</li>
  <li><strong>Three sets of worlds, three jobs.</strong> Screening scores only choose what to look at again. A strategy is kept only on worlds it was not selected on, and it is imitated only in worlds that had no part in keeping it. With 256 equal strategies, 8 worlds and a capture rate near one half, the best one looks about 45 points better than it is {E}.</li>
  <li><strong>The confirmation test.</strong> For each of the W2 worlds, d is the value of the candidate minus the value of the greedy rule in that world. The candidate is kept only if the mean of d exceeds 2.2 times its standard error, which is the standard deviation of d divided by the square root of W2. For four candidates that is a false-pass rate of about 5% {E}. A standing audit sends identical strategies through all three stages; they must pass in at most 5% of cuts {U}.</li>
  <li><strong>Where the trunk continues.</strong> If a strategy was kept: from its end state in one harvest world drawn at random, and half of the time from the greedy clone in one confirmation world instead, so that the rule is also trained on states it reaches by itself. If nothing was kept: from the greedy clone in one confirmation world.</li>
  <li><strong>Exploration noise.</strong> Each strategy draws a temperature between 0.25 and 2, an excursion rate between 0 and 0.3, and a hold length between 1 and 4. At every step a drone either starts an excursion with that rate, which is one random action repeated for up to the hold length, or samples its action from its scores, standardised over its five options and divided by the temperature. Noise is independent between drones, and the rule never sees it, so what search finds is learned as a response to observations alone. The measured recipe contained such held actions {M}; their separate effect was not measured, and version 1 had removed them. The PI has accepted them {D}. The 10 points that were measured belonged to noise shared between drones, which this design does not use. It relies on more clones: before training, 256 independent rollouts found a reward about as often as 64 with shared noise, 7.0% against 9.0% {M}.</li>
  <li><strong>Batteries carry over from leg to leg.</strong> A leg that starts with low batteries cannot be won without a recharger, so recharge-needed legs arise without any task filter.</li>
  <li><strong>Only improvements are stored</strong> {U}. The greedy rule's own captures are not imitated; at the start they are accidents. This differs from the measured recipe, which kept the earliest rewarded rollout even when it was the greedy one. Storing greedy captures with their credit is a comparison arm, judged by how often legs that were solved 50 rounds earlier are still solved.</li>
  <li><strong>A dial that is off at the start:</strong> redrawing the noise of a random block of one to three drones in the best strategies and screening again. Its cost grows in proportion to the number of drones, where sampling all drones independently needs exponentially more rollouts.</li>
  <li><strong>Dropped:</strong> recombination of recorded action lists once anything is hidden, search trees, value estimates, backups, and any search at evaluation time.</li>
</ul>
<h3>Why an untrained team finds reward</h3>
<p>An untrained rule has flat tubes, so drones follow their noise. Against a target that sees two cells, a lone drone cannot capture in the open, so first rewards come from two drones closing by chance and from pockets in the terrain. An untrained team also runs its capturers empty within one battery life, about 12 steps, and an empty capturer stays grounded unless a recharger happens to drive onto its cell. So early trunks hold one short leg, and later legs appear only once capturers have learned to be refilled. How often a first reward occurs is not known and is the first thing measured on the cluster, before any training (§13). The levers, in the order in which they are pulled:</p>
<ol>
  <li>More clones. This is compute and nothing else.</li>
  <li>Smaller grids in the training mixture {D}. With the full team of ten, a new target has a capturer and a second drone within three cells in about 35 to 45% of draws on 12×12, against 4 to 5% on 24×24; with a team of five or six it is about 15% on 12×12. These are hand estimates for drones placed uniformly on a map without obstacles {E}, and being near is not yet a capture. Training maps also keep the dead ends that evaluation maps lose (§3).</li>
  <li>A target that sometimes fails to move, as a declared setting of the environment {Q}. This would reopen the decision that the target is as fast as a drone, for training only.</li>
  <li>Targets with a shorter sight in training {Q}. This makes the task easier in the same way a larger capture radius would, so it comes last.</li>
</ol>
<p>Evaluation always uses the full target. The long evaluation episode, with a recharge, an approach longer than one battery allows, a search and a closure from two sides, cannot be found by sampling in one piece, and training legs end after 48 steps. It has to be assembled by the rule from what short legs taught. Whether that happens is the main risk of the programme.</p>
'''

    # ------------------------------------------------------------------ 8
    out["CREDIT"] = h2(8, "credit", "Credit") + f'''
<p>A capture says that a joint plan worked. It does not say whose behaviour made it work. Two tests answer that for the kept strategy, in each harvest world. Both count captures within the leg limit.</p>
''' + pre([
        "what the drone did                                    simulator only, no lattice",
        ("    for each of R orders of the drones", "an order and its reverse"),
        ("        for j = 0 … n:  replay the leg K times; the first j drones repeat their", ""),
        ("                        recorded actions, the others act at random, the target reacts", "same random draws for every j"),
        ("                        v[j] ← share of the K replays that capture", ""),
        ("        for j = 1 … n:  share of the j-th drone ← v[j] − v[j − 1]", ""),
        "",
        "what the drone sensed                                 with the lattice, same noise, same world",
        ("    run the leg again unchanged, and once per drone with that drone's sensing withheld", ""),
        ("    share of drone i ← capture unchanged − capture without its sensing", "each 1 or 0"),
        "",
        "credit[i] ← the larger of its two mean shares over orders and harvest worlds,",
        "            if that mean clears the threshold; otherwise 0",
    ], 88) + f'''
<ul>
  <li>In a harvest world where the kept strategy did not capture, every share is zero. Such worlds stay in the mean.</li>
  <li>The first test is the rule of the audited 8×8 run {M}, extended from "one drone at random" to random orders. With several capturers, those whose moves mattered share the credit, and one that merely followed gets none. Lesioning one drone at a time in closed loop would give every capturer nothing, because another takes over.</li>
  <li>The second test is the only way an observer can earn credit: in the first test the others repeat their recorded actions, and those actions already contain what the observer found. It becomes a weight in phase 4, where observers first appear, once it has passed the ghost audit there {U}.</li>
  <li><strong>The threshold.</strong> A mean share counts only if it exceeds a multiple of its standard error, taken over the harvest worlds. The multiple is set by the ghost audit: a ghost drone must clear it in at most 2% of legs. With 8 replays, 8 worlds and 2 orders, a threshold of one standard error would let a drone that did nothing through in about one leg in four {E}, so the multiple and the number of replays are expected to rise in phase 3.</li>
  <li>Shares are never divided by a small value and never used per run. A positive weight for drones that did nothing would repeat the failure that was measured when irrelevant motion was imitated: the recharger learned to stand still {M}.</li>
  <li><strong>A second audit.</strong> On teams of up to five, the sampled shares are compared with the exact ones, obtained by averaging over every order of the drones.</li>
  <li><strong>Comparison arm:</strong> closed-loop reruns with drones removed from the simulator, in a random order. One mechanism for both questions, at 9 lattice runs per world for a team of ten.</li>
</ul>
'''

    # ------------------------------------------------------------------ 9
    out["LEARNING"] = h2(9, "learning", "Learning") + pre([
        "for each batch of 64 harvested legs, with iteration counts redrawn",
        ("    rebuild each lattice by replaying its trunk up to the cut", "no gradient"),
        ("    for every step t of the leg", "recorded observations and actions, no noise"),
        "        for every drone i and every j = 1 … 16 with t + j inside the leg",
        "            loss += w(ω, t) · credit[i] · cross-entropy( softmax over the cells of ⟨ Y[present + j], code_i ⟩ ,",
        "                                                         the cell of drone i at time t + j )",
        ("        after every second step: back-propagate through those two steps, update, detach", ""),
    ], 66) + f'''
<ul>
  <li><strong>The weight w(ω, t) follows what the team could know</strong> {Q}. Two harvest worlds are indistinguishable at step t if the team's observations in them have been identical up to t. A dream read at step t in world ω is weighted by the mean return of the kept strategy over the harvest worlds that are indistinguishable from ω at t, and it is imitated in every one of them. While nothing has been sensed, all worlds look alike, so a way of searching is imitated also where it found nothing: it was right there too, and the target only happened to be far away. Once the observations differ, a world stands alone and its dreams are weighted by its own return, so a chase that failed is not imitated. The rule is stated on observations and names nothing in the task.</li>
  <li>With everything visible, every world stands alone from the first step, and the rule reduces to imitating the kept strategy's captured legs. The measured system did the same, except that it also kept the greedy rollout when that was the earliest rewarded one.</li>
  <li>Each drone has one softmax over the cells of a slice: a drone is in one place at a time. There is no shared target per kind, which could not tell two drones on one strand from two drones on two.</li>
  <li>Except in the first steps of a trunk, the loss starts from a state that has already run for hundreds of iterations and carries no gradient. Training on its own long-run states is what the published recipes for "more iterations must not hurt" have in common; here it comes with the kept lattice. Iteration counts are redrawn for every replay (K0 from 32 to 128, k_s from 2 to 6), and Gaussian noise of standard deviation 0.02 is added to every state update during replay.</li>
  <li><strong>What is stored for a leg:</strong> the seeds of map, team and worlds, the trunk's actions up to the cut, the leg's actions in each harvest world, the codes, the credits and the weights. The learner runs the simulator again from these to regenerate the observations. Legs are used in the round that found them. Optimiser Adam, learning rate from 0.001 falling to 0.0002.</li>
  <li>There is no forecast loss, and no gradient or selection signal ever comes from the hidden position of the target. The diagnostic readout of §10 is fitted afterwards on saved states, in a separate process whose code the learner does not import.</li>
</ul>
'''

    # ------------------------------------------------------------------ 10
    out["GATES"] = h2(10, "gates", "Gates and controls") + f'''
<p>A gate is a check that every later phase must keep passing. A control is the same model on the same seeds with one thing taken away.</p>
''' + table(["Gate", "What must hold", "The 8×8 rule today"], [
        ["More computation never hurts (the iteration gate)", "K0 and k_s are multiplied together by 2, 4 and 8. On 2,000 held-out tasks the lower 95% bound of the paired difference in closed-loop success stays above −2 points. A longer window and more clones are checked by retraining at the end of phases 3 and 5", f"Open loop, at twice the iterations the two checkpoints lost 5 and 18 points {M}"],
        ["One drone, one strand", "On 8×8 with one recharger and two, three or five seekers that each have their own target, all seekers arrive in at least 60% of tasks", f"34% for two seekers in one channel, never trained that way {M}"],
        ["Tubes, not arrows", "With everything visible, clamping and iterating only every fourth step, while drones follow the tubes already grown, costs at most 5 points. The share of steps in which a drone's planned cell two steps ahead changes although no new fact arrived is reported", f"Flying the whole 8×8 dream without replanning cost 2 to 3 points {M}"],
        ["Reach", "Capture rate falls by at most 10 points between episodes whose first sighting is within 10 cells of the nearest capturer and those where it is 30 to 40 cells away, and doubling K0 does not improve it", "Not applicable"],
        ["Search does not keep luck", "Identical strategies sent through screen, confirm and harvest pass confirmation in at most 5% of cuts", "Not applicable"],
        ["Credit does not pay bystanders", "A ghost drone clears the credit threshold in at most 2% of legs", "Not applicable"],
        ["No domain concept in the learner", "A text search of the learner's source finds none of the words in rule 1 of §4, and the learner imports no benchmark code. Both checks are part of the test suite on the cluster", "Passed in the audited path"],
        ["The evaluator is independent", "A second implementation of simulator and readout, written from this document, reproduces the stored numbers", f"Done once, for the first passing checkpoint {M}"],
    ]) + table(["Control", "Certifies", "Expected"], [
        ["Rechargers removed, or left in the team with refills switched off", "A recharge is required", "Exactly 0. Any capture is a bug in the benchmark"],
        ["Rechargers frozen where they start", "Recharging must be organised, not merely available", "At most 5%"],
        ["One capturer alone", "Coordination is required", "Exactly 0"],
        ["One capturer with all rechargers and observers", "How much capture comes from herding by drones that cannot capture", "Reported"],
        ["Obstacle channel zeroed in the input", "Obstacles must be flown around", "At most half the headline rate"],
        ["Observers removed; observers flying with their sensing withheld", "Scouting matters, and through information", "A drop whose interval excludes zero"],
        ["Scripted team that always sees the target", "Capture is possible on the evaluation set", "At least 90%, published as a reference"],
        ["Target always visible", "The price of partial observability", "Reported"],
        ["Batteries unlimited", "The price of energy", "Reported"],
        ["A readout of the target's cell, fitted afterwards on saved states", "The state still holds where the target was after the sighting has left the window", "Better than 'where it was last seen', eight steps later"],
        ["Random team", "The task is not solved by chance", "At most 1%"],
    ]) + '''
<p>Logged in every training run, because each is the first sign of a predicted failure: the share of cuts in which any strategy captures and the share in which one is confirmed; the distance from the new target to the nearest capturer in harvested legs against all legs; steps survived per capturer and legs per trunk; credit per kind; recharges per captured leg; how many drones the target saw at the moment of capture and whether the capturing drone was moving towards it; the size of the state; how often two drones of one kind share a cell.</p>
'''

    # ------------------------------------------------------------------ 11
    out["COMPUTE"] = h2(11, "compute", "Compute") + table(["Dial", "Start", "More of it buys", "Paid in"], [
        ["Trunks and cuts per round", "32 trunks, about 64 cuts", "More situations per update", "Everything below"],
        ["Strategies per cut, M", "32", "More chances of a better leg", "Lattice runs"],
        ["Worlds: screen W1, confirm W2, harvest W3; candidates E", "W1 = 4, E = 4, W2 = 16, W3 = 8", "Selection by skill instead of luck", "Lattice runs"],
        ["Leg limit H; preference for speed γ", "48 steps; 0.97", "Longer chases can be credited", "Lattice runs"],
        ["Iterations at reset K0, per step k_s", "48, 4", "Reach, and computation per decision", "K0 once per trunk; k_s in every clone"],
        ["Window: past, future", "3, 16", "Memory of facts, depth of the dream", "Every iteration"],
        ["State channels, hidden units", "32, 128", "Capacity of the rule", "Every iteration"],
        ["Credit: replays K, orders R", "8; 2", "Sharper shares", "Simulator steps only"],
        ["Block redraws of the best strategies", "Off", "Joint changes at a cost linear in the team", "Lattice runs"],
        ["Commit groups", "1", "Fewer drones on one strand", "One extra iteration per group"],
        ["Back-propagation span", "2 steps", "Memory that is taught over longer gaps", "Training memory"],
        ["Batch; learning rate", "64 legs; 0.001 falling to 0.0002", "Steadier updates", "Learning time"],
    ]) + f'''
<h3>Cost model</h3>
<p>All figures are hand arithmetic on a throughput that has not been measured {E}. The first act on the cluster is to measure it.</p>
<ul>
  <li>Window at 24×24: 20 slices × 576 cells = 11,520 voxels. One iteration costs 131,000 multiply-adds per voxel. The input convolution runs once per step and costs 1.8 iterations.</li>
  <li>One clone-step: 5.8 iterations × 11,520 voxels = 67,000 voxel-iterations. Version 1, with 12 iterations on its 27-slice window and a slightly larger rule, needed about 207,000 in the same unit, three times more.</li>
  <li>One cut: 32 × 4 + 5 × 16 + 8 = 216 clones of at most 48 steps: 10,400 clone-steps, or 6.9 × 10<sup>8</sup> voxel-iterations. Clones start from the trunk's lattice, so K0 is paid once per trunk.</li>
  <li>One round: search at 64 cuts, 4.4 × 10<sup>10</sup>; the sensing test, 11 reruns in each of 8 harvest worlds for every kept strategy, up to 1.8 × 10<sup>10</sup>; rebuilding trunks and learning, about 0.3 × 10<sup>10</sup>. Together about 6 × 10<sup>10</sup> voxel-iterations when half of the cuts keep a strategy. Version 1 at this scale and at its own dials: about 2 × 10<sup>12</sup>.</li>
  <li>At an assumed 110 to 440 million voxel-iterations per second per H100 (the earlier estimate of 100 to 400 million, rescaled to this rule): two to nine minutes per round on one GPU, and <strong>35 to 140 GPU-hours for a 900-round run at full scale</strong>. On seven H100s, the number assumed until the cluster is known, that is 5 to 20 hours. A 12×12 run costs about a quarter of that or less, a 16×16 run a little under half.</li>
  <li>The 900 rounds are carried over from the 8×8 runs. Under a hidden target the improvement per round is smaller, and the number of rounds may be several times higher. Budgets are set again after the throughput benchmark and a 25-round take-off test.</li>
</ul>
'''

    # ------------------------------------------------------------------ 12
    out["PRIORS"] = h2(12, "priors", "Priors and removals") + '''
<p>What the design still assumes, why, and the test that could remove each assumption. What was removed since version 1 is listed in §1.</p>
''' + table(["Prior", "Why it is there", "Test that could remove it", "Status"], [
        ["A local rule shared by all voxels of a space-time lattice, with tubes as output", "The object of study", "None", "Kept"],
        ["3×3×3 neighbourhood with zero padding", "The smallest symmetric neighbourhood; padding makes the edges of the map and of the window visible", "A wider neighbourhood at equal compute", "Kept"],
        ["One softmax per drone and slice; a one-step peek maps an action to a cell", "A drone is in one place; the readout must know where an action leads", "A per-voxel loss; scores read at the drone's own voxel", "Kept"],
        ["Codes that give each drone its own channel for an episode, at most 16 drones", "Binding without fixed roles and without matching", "One channel per kind, trained with several drones in it; codes as random rotations", "Kept"],
        ["Kinds as three fixed labels; the map as input", "Inherited from the 8×8 design, where each type had its own channel", "Zero each channel in the controls", "Kept"],
        ["Each drone's battery level as input", "The one permitted exception", "The 8×8 design, which hid it", D],
        ["All sensing pooled in one lattice at once", "Decided", "A communication range", D],
        ["Exploration noise held over several steps", f"Part of the measured recipe {M}. Version 1 had removed it", "Single-step noise at equal clones", D],
        ["Legs with a 48-step limit; return γ^(steps), γ = 0.97, counted to the end of the leg", "Respawning targets; sooner capture preferred. A count from the simulator, not a learned value", "Whole-episode selection at equal budget; capture within the limit, without γ", D],
        ["Imitation weighted by the mean return over indistinguishable worlds, including worlds without a capture", "The only way a way of searching can be imitated without a shaped reward", "Own-world return on every step", Q],
        ["The simulator can be saved, restored, redrawn in what is hidden, replayed with drones at random, and run with a drone's sensing withheld", "Trunks, cuts and credit need it", "Greedy-only trunks; no credit", D],
        ["Training mixture over grid sides 12, 16, 24 and team sizes", "Dense encounters for an untrained team; size generalisation comes from varied sizes", "Train on the full scenario only", D],
        ["Training maps keep dead ends; evaluation maps lose them", "Terrain may help an untrained team; it must not do the capturing in the test", "No dead ends in training either", Q],
        ["Benchmark-side connectivity checks, shortest-path detour measure, two scripted teams, a readout of the hidden target position fitted afterwards", "Certify the properties of §3 and measure belief, outside the learner", "Certify by controls alone", Q],
        ["Commit groups; block redraws of one to three drones", "Dials for binding and for joint changes, off at the start", "One group at equal iterations; no redraws at equal clones", "Kept as dials"],
        ["A target that sometimes fails to move, or sees less far, in training", "Only if an untrained team finds no reward otherwise", "The full target from the first round", f"Not in the design. {Q} before use"],
        ["A noise field the lattice can see, or noise shared between drones", "Only if joint moves are sampled too rarely", "Independent readout noise at equal clones", f"Not in the design. {Q} before use"],
        ["Coarser copies of the lattice at half and quarter resolution", "Only if the reach gate fails", "The flat lattice with more iterations at equal cost", f"Not in the design. {Q} before use"],
        ["A forecast of the target trained on later sightings; state channels that do not slide with the window", "Only if what was seen is forgotten", "The design as it stands", f"Not in the design. {Q} before use"],
    ])

    # ------------------------------------------------------------------ 13
    out["ROADMAP"] = h2(13, "roadmap", "Roadmap") + f'''
<p>Seven phases. Each is a short series of small experiments with one change per run and one gate. A phase that fails its gate is analysed on trajectories and then repeated or stopped; nothing outside the dials of §11 is added to the learner to get past a gate. The rows of §15 are decided before the phase that needs them: the benchmark rows before phase 1, the search and training rows before phase 2. Phases 2 to 5 are development steps. The claim in phase 6 is made by one recipe trained from scratch. GPU-hours are estimates {E} for the number of runs named in each row, at 900 rounds per run.</p>
''' + table(["Phase", "Goal", "Experiments", "Gate", "GPU-hours"], [
        ["0 · Arrival", "Replace every estimate the plan rests on. No learner",
         "Connect; probe the cluster, including git push rights; throughput in three numeric formats; restart drill; the existing tests on the cluster's CPUs; re-evaluate the stored 8×8 checkpoint",
         "Throughput recorded and the cost model redone. The stored checkpoint reproduces 1,507 of 2,000", "under 5"],
        ["1 · Benchmark", "A frozen benchmark whose properties are shown. No learner",
         "Tensor simulator with tests against hand-worked cases and exact replays; map generators; the two scripted teams; calibration of batteries, radii and horizon; enumeration of small games to check the hand proofs; captures by an untrained team per grid size and team, split by whether the capturing drone had sensed the target and was moving",
         "Scripted team captures at least 90%. One capturer: 0. Rechargers removed: 0; frozen: at most 5%. Observers' sensing is worth at least 10 points to the second scripted team. Random team at most 1%. Parameters, seeds and set hashes committed. Kill: no setting meets these, and the values go back to the PI", "about 10"],
        ["2 · Core on 8×8", "The new lattice loses nothing on the measured task",
         "First the old recipe unchanged on the cluster, with and without shared noise. Then one change per run: window with kept and bounded state, codes, battery input, staged search, credit orders. Arms: state without a bound, regrowing at every step, one channel per kind, greedy captures stored. Three seeds each, about 20 runs",
         "The port's three-seed means lie within 5 points of the stored runs (75.4% with shared noise, 65.5% without). The new core, which has no shared noise, reaches at least 75% on never-posed tasks, passes the iteration gate and the strand gate. Kill: no arm passes the iteration gate", "25 to 95"],
        ["3 · Pursuit", "Coordinated capture of a visible fleeing target on 12×12: 2 to 5 capturers, obstacles, roads, batteries, rechargers",
         "The credit forms against the exact shares; one, two and n commit groups at equal iterations; block redraws on and off; staged search against no selection; the audit with identical strategies. About 12 runs",
         "At least 60% with three capturers; one capturer alone exactly 0 on maps without dead ends. No kind with zero mean credit; the ghost drone passes in at most 2% of legs. Kill: under 30% after 600 rounds with every lever of §7 that needs no approval", "70 to 300"],
        ["4 · Hidden target", "Scouting through the value of information: observers, 16×16, six drones",
         "Target visible, hidden with observers, hidden without; observers with sensing withheld; cut before against after the draw; number of worlds; past window 3, 8, 16 and back-propagation span 2, 8, 16; the readout of the saved state. About 9 runs",
         "Hidden reaches three quarters of visible. Withholding the observers' sensing costs at least 10 points. Kept strategies stay ahead on a further set of worlds. Kill: no observer effect; the question then goes back to the PI, and no scouting bonus is added", "140 to 560"],
        ["5 · Full scenario", "24×24, ten drones, from scratch",
         "Three seeds of the full recipe; the reach gate; all controls; review of 50 failed trajectories per run",
         "At least 80% on validation, as a point estimate. Reach gate passed. Frozen rechargers at most 5%. Kill: under 40% after three full runs while the visible-target rate is at least 90%; the failed gate is reported and no heuristic is added", "105 to 420"],
        ["6 · Certification", "A number that survives an audit",
         "Protocol and set hashes committed first. Five seeds from scratch. 2,000 test episodes, the held-out map family, the hand-drawn maps, other flee rules, every control on identical seeds, compute sweeps, an independent re-implementation of the evaluator",
         "The bar of §3 on the test set, with every control behaving as stated", "175 to 700"],
    ], cls="wide phase") + '''
<p>Together that is roughly 500 to 2,100 GPU-hours if 900 rounds per run suffice, which is 3 to 12 days of computing on seven H100s, spread over several weeks of experiments and analysis. Evaluation, controls and sweeps add about a tenth. Phases 5 and 6 grow in proportion to the number of rounds. Each run is sized to finish within hours.</p>
<h3>Predicted failures and what they look like first</h3>
''' + table(["Failure", "First sign", "Response"], [
        ["No reward for an untrained team at full scale", "Fewer than 10 captured legs per 4,096 untrained clones in phase 1", "The levers of §7 in order; the last two need approval"],
        ["The team learns to wait for the target to walk into it", "Captured legs in which the capturing drone never moved towards the target", "Count such legs before trusting any early curve; nothing greedy is stored"],
        ["Capturers chase one by one and never close from two sides", "Harvested legs start within two cells of a capturer and stay there over rounds; one drone in the target's sight at capture", "Block redraws and more strategies; two commit groups"],
        ["Search keeps luck", "The audit passes identical strategies; kept strategies do not stay ahead on a further set of worlds", "More confirmation worlds, fewer strategies, at equal cost"],
        ["The kept state drifts on long trunks", "State pinned at its bound; capture rate falls with the number of the leg; plans change without new facts", "More update noise; iteration counts redrawn more widely"],
        ["Noisy credit brings back 'stand still'", "Share of stay actions per kind rises; the ghost drone clears the threshold too often", "More replays and orders; a higher threshold"],
        ["Observers add nothing, or only herd", "Withholding their sensing changes nothing", "Check the value of information in the benchmark first; more harvest worlds. Never a coverage bonus"],
        ["What was seen is forgotten once it leaves the window", "Capturers return to cells just found empty; the readout loses the target a few steps after the sighting", "Longer back-propagation span, longer past window, wider state"],
        ["The reach gate fails", "A sharp drop of capture rate with the distance of the first sighting", "More iterations per step. Then coarser lattice levels, which need approval"],
        ["Throughput far below the estimate", "The arrival benchmark", "Narrower rule and smaller dials; every budget rescaled before phase 2"],
    ])

    # ------------------------------------------------------------------ 14
    out["CLUSTER"] = h2(14, "cluster", "Working on the cluster") + f'''
<p>No project code runs anywhere but on the H100 cluster and its CPUs {D}. That covers training, evaluation, unit tests, shape checks, benchmarks, analysis scripts and the drawing of result figures. The laptop holds the git working copy and is where ssh and git commands are issued, by a session attached to that folder; it runs nothing else. The assistant's cloud container edits text, builds this document and reads results; it executes no project code.</p>
<ol>
  <li><strong>Code moves by git only.</strong> A change is committed and pushed. On the cluster the exact commit is checked out into the directory of the run that uses it, so a run can always be traced to its source.</li>
  <li><strong>Verification comes first and also runs on the cluster.</strong> Every increment starts with the test suite on the cluster's CPUs: simulator rules against hand-worked cases, tensor shapes, replays that must reproduce recorded episodes exactly, the text search and import check of §10, and the controls of §10 on the scripted teams.</li>
  <li><strong>Jobs never depend on the connection.</strong> They are submitted to the scheduler and outlive the ssh session. Each job writes a small status file every minute (time, round, metrics, restart count), appends to a log, saves a checkpoint every 5 to 10 minutes by writing and renaming, and resumes from the latest checkpoint when it is restarted.</li>
  <li><strong>Reading results costs one ssh call.</strong> Status files, metric tables and text renderings of trajectories are read directly. Figures and trajectory files that should be kept are committed on the cluster and come back through git, which needs push rights there.</li>
  <li><strong>Every experiment has a card in the repository</strong> before it starts: the hypothesis, what would count as support and as refutation, the commit, the configuration, and afterwards the result and the decision taken. One experiment is one committed configuration and at most a few hours on the available GPUs. Failure analysis looks at trajectories, not at one number.</li>
  <li><strong>Search and learning are separate processes,</strong> one search worker per GPU and one learner. Workers read weights from a file the learner replaces atomically, and write harvested legs as small records (§9). This survives restarts and scales with the number of GPUs.</li>
</ol>
<p>Two files for the first session exist and have not been executed anywhere: <code>nca_drones/cluster/probe.sh</code> reports what the cluster offers without changing anything, and <code>nca_drones/cluster/bench_rule.py</code> measures the rule's throughput in three numeric formats.</p>
'''

    # ------------------------------------------------------------------ 15
    out["DECISIONS"] = h2(15, "decisions", "Open decisions") + '''
<p>Each row is a choice that changes what the result means or departs from "general mechanisms only". The proposal is the recommended choice. Nothing that depends on a row is built before the PI has decided it; the last column names the phase that waits for it.</p>
''' + table(["Decision", "Options", "Proposal and reason", "Before"], [
        ["Cluster and connection", "Session attached to the laptop folder, ssh from there; or a session on the laptop", "Chosen by the PI: the Mac folder is linked to this session and ssh goes from there. Still needed: the folder added in the desktop app, the ssh alias, scheduler, number of H100s and for how long, working directory, whether the cluster reaches GitHub and PyPI and may push", "Phase 0"],
        ["Tools on the benchmark side", "Allowed outside the learner; or certify by controls alone", "Allowed: connectivity check; shortest-path detour measure; a scripted team that always sees the target, for admission and as a reference; a second one that acts on what was sensed, for calibrating radii; enumeration of small games; a readout of the target's cell fitted afterwards. The learner imports none of it", "Phase 1"],
        ["Flee and wander rule, sight, metric", "The rule of §3; a rule that looks further ahead; sight blocked by obstacles", "§3: one step ahead, greatest distance to the nearest drone seen, staying allowed, two cells of sight through obstacles, Manhattan distances. Blocked sight would let one capturer ambush and would void the coordination proof", "Phase 1"],
        ["A drone with an empty battery", "Decided by the PI: it cannot fly, and it flies again once a recharger on its cell has refilled it. Open: what it can still do while empty", "While empty it cannot sense or capture, and the target ignores it. An empty capturer that could still capture would void the recharge certificate", "Phase 1"],
        ["How 'requires a recharge' is certified", "Target farther than twice each capturer's initial charge; or a target that appears after one battery life; or the frozen-recharger control alone", "Distance against initial charge. It is arithmetic, it forces the recharge through pursuit instead of waiting, and it covers the gathered start with full batteries as well as the spread start with random levels", "Phase 1"],
        ["What counts as coordination; dead ends", "Any two drones in the target's sight, or two capturers. Dead ends removed in evaluation only, or in training too", "Any two drones: it follows from 'flees any drone it sees', and results are reported split by who was in sight. Dead ends removed in evaluation only", "Phase 1"],
        ["A training target that is not captured", "Moves to a new random cell after 48 steps; or stays until the episode is reset", "Moves. Otherwise one unreachable target blocks an episode", "Phase 1"],
        ["Starting values", "Sight and sensing 2 / 3 / 5 / 2 cells; batteries 12 and 24; horizon up to 128; at most 16 drones", "Calibrated in phase 1 without a learner, then frozen", "End of phase 1"],
        ["The bar", "Point estimate of 80%, or lower 95% bound at 80%; one training seed or five; held-out maps inside the bar or beside it", "§3: lower bound at 80% over five seeds on new maps of the training families; held-out family and hand-drawn maps beside it at 60%. The 8×8 run met its 75% bar as a point estimate (82.0% on 200 tasks) and missed it as a lower bound (73.4% on 2,000)", "End of phase 1"],
        ["Changing the benchmark after training has started", "Only with approval, as a new version on which earlier results are reported again; or never", "Only with approval", "Phase 2"],
        ["Search from saved simulator states", "Cuts, trunks that continue from kept clones, replays with drones at random, runs with a drone's sensing withheld; or none of these", "Accepted by the PI. It is rollout search with returns counted in the simulator: no value table, no backup, no knowledge of the task", "Decided"],
        ["Unit of selection and preference for speed", "Leg with γ = 0.97; capture within the leg limit without γ; whole episode", "Accepted by the PI: leg with γ. The 8×8 runs kept the earliest rewarded rollout, which ranks plans the same way", "Decided"],
        ["Exploration noise held over several steps", "Allowed as a listed prior with a single-step arm; or single-step noise only", "Accepted by the PI: it is in the measured recipe", "Decided"],
        ["Training mixture", "Grid sides 12, 16, 24 and varied teams, uniformly; or the full scenario only", "Accepted by the PI: the mixture", "Decided"],
        ["Imitation weights under a hidden target", "Mean return over indistinguishable worlds, failures included; or own-world return only", "Mean over indistinguishable worlds: without it a way of searching earns nothing where it found nothing", "Phase 4"],
        ["Levers held back", "A target that sometimes fails to move or sees less far (reopens the decision on target speed, for training); a noise field the lattice can see, or shared noise; coarser lattice levels; a forecast of the target, or state that does not slide", "None is in the design. Each would be asked for separately, with the measurement that makes the case", "When a gate fails"],
    ]) + '''
<footer>
  <p>Measured numbers come from the 8×8 experiments in the repository (<code>nca_drones/README.md</code>, <code>nca_drones/results/</code>). This version was written after a literature sweep, a panel of five independent designs with adversarial reviews and a judge, and a second review of this text. Most literature claims could not be checked against the primary source, and none of them is relied on for a number here. Version 1 of this page is kept in the repository as <code>nca_drones/docs/unified_wavefront_design.html</code>.</p>
</footer>
'''
    return out
