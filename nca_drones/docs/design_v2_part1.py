"""Text of version 2 of the design page, sections 1 to 6. Kept apart from the builder so that sections can be edited one at a time."""

M = '<span class="tag m">measured</span>'
U = '<span class="tag u">not yet tested</span>'
E = '<span class="tag u">estimate</span>'
D = '<span class="tag d">decided</span>'
Q = '<span class="tag q">needs approval</span>'

TOC = [
    ("changes", "What changed since version 1"),
    ("forces", "What the task forces"),
    ("benchmark", "The benchmark"),
    ("contract", "The bitter-lesson contract"),
    ("lattice", "The lattice"),
    ("acting", "Acting"),
    ("search", "Search"),
    ("credit", "Credit"),
    ("learning", "Learning"),
    ("gates", "Gates and controls"),
    ("compute", "Compute"),
    ("priors", "Priors and removals"),
    ("roadmap", "Roadmap"),
    ("cluster", "Working on the cluster"),
    ("decisions", "Open decisions"),
]

TERMS = [
    ("Lattice", "The grid of voxels with axes time, x and y: one slice per time step, one voxel per map cell."),
    ("Rule", "The one local update that is applied to every voxel at every iteration."),
    ("Tube", "For one drone, the marks in future slices that say where it will be. A tube seen as a path is a strand."),
    ("Front, dream", "The present slice; the slices ahead of it, which the rule fills in."),
    ("Clamp", "To overwrite the input channels of a slice with what is known. State and output are never written from outside."),
    ("Code", "A one-hot vector of length 16 that a drone receives for one episode. It selects that drone's channel."),
    ("Greedy rule", "The rule acting without exploration noise."),
    ("Strategy", "The greedy rule plus one stream of exploration noise."),
    ("World", "A seed for everything the team cannot know: where the target appears and how it breaks ties."),
    ("Clone", "One strategy run in one world from a saved state."),
    ("Trunk, cut, leg", "A training episode; a saved state just before a target is drawn; the stretch from a cut to the capture of that target or to the leg limit."),
    ("Screen, confirm, harvest", "The three stages of search at a cut (§7)."),
    ("Ghost drone", "A drone with a code, a cell and a tube that the simulator ignores: it cannot sense, capture or be seen. It is used to audit credit."),
    ("Scripted teams", "Two hand-written reference teams that exist only on the benchmark side (§3)."),
    ("Free cell", "Any cell that is not an obstacle. Road cells are free cells."),
]

HEADER = f'''<header>
  <p class="eyebrow">Specification, version 2 · space-time cellular automaton for multi-drone pursuit</p>
  <h1>Unified Wavefront Design</h1>
  <p class="lede">Version 2 rebuilds the design from first principles for the target the project now has: a 24×24 map with obstacles and roads, ten drones of three kinds, and an evasive target that stays hidden until a sensor finds it. One lattice, one local rule and tubes as the only output that steers remain. Most of what version 1 added around them is removed.</p>
  <div class="status"><span class="label">Status</span><p>Nothing in this document has been run. The simulator of §3 is written but not yet executed; nothing else is built. {M} rests on the earlier 8×8 experiments. {D} was fixed by the PI at kickoff. {Q} touches what the benchmark means, or departs from "general mechanisms only"; nothing that depends on such an item is built before the PI has decided it (§15). {E} is replaced by a cluster measurement before it is used.</p></div>
  <div class="terms"><h3>Terms</h3><dl>
''' + "\n".join(f"    <dt>{t}</dt><dd>{d}</dd>" for t, d in TERMS) + '''
  </dl></div>'''


def table(head, rows, cls="wide", num=()):
    th = "".join(f'<th{" class=num" if i in num else ""}>{h}</th>' for i, h in enumerate(head))
    body = "\n".join("    <tr>" + "".join(f'<td{" class=num" if i in num else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}">\n  <thead><tr>{th}</tr></thead>\n  <tbody>\n{body}\n  </tbody>\n</table></div>'


def h2(n, sid, title):
    return f'<h2 id="{sid}"><span class="sec">§{n}</span>{title}</h2>'


def sections(b):
    S = []

    # ------------------------------------------------------------------ 1
    S.append(h2(1, "changes", "What changed since version 1") + '''
<p>Version 1 was written for two drones on an 8×8 grid and then stretched. At the new scale several of its parts fail on cost or on logic, and one input changed by decision.</p>
''' + table(["Version 1", "Version 2", "Reason"], [
        ["Drones commit one after another; the rule runs after each", "All drones read their own tube at once",
         "Ten drones cost 12 passes of the whole lattice per step. Sequential commitment was there to break ties between drones of one type, which is now the job of codes (§5)"],
        ["One tube channel per drone type", "One code per drone; a tube is read by an inner product with that code",
         f"Two seekers placed in one channel both arrived in 34% of cases, each alone in 81.5% {M}. That rule had never been trained with two seekers, so a rule trained with shared channels is a comparison arm"],
        ["The battery is never shown", "Each drone's battery level is an input", f"The one permitted exception {D}"],
        ["Best strategy over a grid of strategies and worlds, kept if it beats the greedy one", "Search inside one leg: screened on shared worlds, kept only if confirmed against the greedy rule on fresh worlds, imitated on a third set",
         f"With 256 strategies and 8 worlds the old test passes by luck most of the time when nothing is better {E}"],
        ["Recorded action lists of different clones are recombined", "Dropped whenever anything is hidden",
         "A recorded list carries the hidden information of the world it came from"],
        ["Each drone is lesioned alone in closed-loop lattice runs", "Credit from simulator-only replays over random orders; a rerun per drone with its sensing withheld",
         f"Lesioning one drone at a time gives every capturer nothing when another takes over. The simulator-only form is the one the audited 8×8 run used {M}"],
        ["Episodes are the unit of selection and imitation", "Legs: from the appearance of a target to its capture",
         "Targets respawn in training, so an episode has no single reward to assign"],
        ["Forecast outputs and a forecast loss", "Off. First thing to add back if scouting does not appear",
         "Not forced by the task; its training signal is biased towards cells that were visited"],
        ["Window of 6 past and 20 future slices", "3 past and 16 future slices", "Cost: a quarter fewer voxels. The dream still reaches further than one capturer battery"],
        ["State without a bound", "State bounded per voxel",
         f"At twice the trained iterations the two 8×8 checkpoints lost 5 and 18 points; at one and a half times, one lost 8.5 and the other nothing {M}. The cause is not established. A bound is the hypothesis, tested first {U}"],
    ]))

    # ------------------------------------------------------------------ 2
    S.append(h2(2, "forces", "What the task forces") + '''
<p>Each mechanism in this design answers one requirement that the task imposes on any learner restricted to reward, a simulator, search and learning. A mechanism without such a requirement is not in the design.</p>
''' + table(["Requirement", "Why it cannot be avoided", "Mechanism"], [
        ["Reward must occur under an untrained rule", f"Imitation has nothing to copy before the first capture. On 8×8 an untrained rule found a rewarded rollout on 1.2% of tasks with independent noise and on 9.1% with shared noise {M}",
         "Exploration noise that makes drones travel; respawning targets; smaller grids and teams in the training mixture (§7)"],
        ["A tube must say which drone it belongs to", "Five capturers share a kind, and a pincer sends two of them to two different cells", "Codes (§5)"],
        ["Motion that did not matter must not be imitated", f"Without credit the recharger learned to stand still {M}. With ten drones most are irrelevant to any one capture", "Credit (§8)"],
        ["Luck must not be selected", "With a hidden target, the best single rollout is the one that happened to fly towards it", "Shared worlds, confirmation on fresh worlds, imitation on a third set (§7)"],
        ["An informer must be creditable", "An observer never captures, and a replay of recorded actions already contains what it found", "A rerun with that drone's sensing withheld (§8)"],
        ["The plan must persist and survive more computation", f"Regrowing the lattice at every step gained 22 points on hard starts and lost 5 on easy ones by flipping between plans {M}. On 24×24 it would also cost a full growth per step", "A bounded state that is kept and trained on its own long-run states (§5, §9). Regrowing stays a comparison arm"],
        ["Cost per decision must not grow with the team", "Version 1 needs one lattice pass per drone per step", "Simultaneous readout (§6)"],
        ["Information must cross the map faster than drones fly", "A 3×3×3 rule moves information one cell per iteration; routes on a 24×24 map with detours are 50 to 100 cells long", "Iterations at reset and at every step, checked by the reach gate (§10)"],
        ["Training must be cut into units that end in a reward", "Targets respawn, batteries carry over, episodes have no natural end", "Legs (§7)"],
    ]))

    # ------------------------------------------------------------------ 3
    S.append(h2(3, "benchmark", "The benchmark") + f'''
<p>The benchmark is part of the claim. "80% capture" has no meaning until the definitions below are frozen. Values marked {E} are frozen at the end of phase 1 from runs that involve no learner (§13).</p>
<figure>
  <div class="scroll diagram">{b.fig_map()}</div>
  <figcaption>Who may go where, on an excerpt of a map. A capturer has to fly around an obstacle; the observer flies straight over one. The recharger cannot leave the road, so every refill happens on a road cell. The target moves like a capturer and starts to flee once any drone is within two cells.</figcaption>
</figure>
''' + table(["Kind", "Number in the hardest scenario", "May enter", "Battery", "Senses", "Can"], [
        ["Capturer", f"5 {D}", f"Free cells {D}", f"Limited {D}; 12 steps {E}", f"3 cells {E}", "Capture"],
        ["Observer", f"2 {D}", f"Every cell, obstacles included {D}", f"Limited {D}; 24 steps {E}", f"5 cells {E}", "Nothing else"],
        ["Recharger", f"3 {D}", f"Road cells only {D}", f"Unlimited {D}", f"2 cells {E}", "Refill every drone on its cell"],
        ["Target", "1 at a time", f"Free cells {D}", "None", f"Drones within 2 cells {E}", "Flee"],
    ]) + f'''
<h3>Rules</h3>
<ul>
  <li><strong>Moves.</strong> Stay, north, south, west, east. Everyone moves one cell per step {D}. A move into a cell the kind may not enter becomes a stay.</li>
  <li><strong>Order within a step.</strong> (1) Drones and target choose from the same state. (2) All move at once. (3) Capture is tested. (4) Every battery loses one unit, hovering included; a drone on a recharger's cell is then refilled; a drone at zero is grounded {D}: it stays where it is and cannot fly until a recharger comes to its cell and refills it. While it is empty it also cannot sense or capture, and the target ignores it {Q}. A capture on the last unit counts, and a drone that arrives on a recharger's cell with its last unit keeps flying. (5) Sensing. (6) In training, a new target is drawn if there is none.</li>
  <li><strong>Capture.</strong> A capturer and the target are on the same cell after the move, or they exchanged cells during it {D}. A capture radius exists as a setting and is zero {D}. An evaluation episode ends at the capture.</li>
  <li><strong>Target.</strong> It flees from any drone it can see and wanders otherwise {D}. Proposed details {Q}: it sees every drone, of any kind, within two cells, through obstacles. If it sees none, it stays or steps to a free neighbour, at random. Otherwise it takes the option, staying included, that maximises its distance to the nearest drone it sees, measured to where the drones stood before the move, ties at random.</li>
  <li><strong>Information.</strong> All drones feed one lattice. The map is known. The target is known only inside some drone's sensing radius {D}. A sensed cell reports "target here" or "sensed, empty". All distances are Manhattan distances, and no drone's sensing is blocked by obstacles {Q}.</li>
  <li><strong>Reward.</strong> One per capture. Nothing else.</li>
</ul>
<p>Sensing radii are chosen so that a capturer sees the target one cell before the target sees it, and an observer can shadow it from three to five cells away without making it flee. That is what gives scouting a value.</p>
<h3>Maps</h3>
<p>Roads are laid first: junctions on a jittered coarse lattice, joined by one-cell-wide segments; a spanning tree is kept and a share of the remaining segments is added back to make loops; dead-end spurs are attached. Obstacles fill part of the space between roads. Three generator families are used in training (city blocks with alleys, scattered walls, braided mazes). A fourth family and twenty hand-drawn maps are never trained on. Every map must have connected free space, connected roads, and a road within four steps of every free cell {E}.</p>
<p>In evaluation maps, a free cell with fewer than two free neighbours is turned into an obstacle, repeatedly until none is left, so that a dead end cannot do the capturing. Where such a cell is a road cell, a neighbouring obstacle is opened instead and the road stays. Training maps are left as generated {Q}.</p>
<h3>Training and evaluation</h3>
<ul>
  <li><strong>Training.</strong> After a capture the next target appears on a random free cell with no drone on it. There is never more than one {D}. Battery levels, empty drones included, carry over until the episode is reset after 96 steps {E}. A target that is not captured within 48 steps moves to a new random cell {Q}. Maps, grid sides of 12, 16 and 24, and teams of 1 to 5 capturers, 0 to 3 rechargers and 0 to 2 observers are drawn uniformly, without stages {D}. Drones start on random cells their kind may enter, with battery levels drawn uniformly between 2 units and full.</li>
  <li><strong>Headline evaluation.</strong> A 24×24 map never seen in training, with the full team of ten. Half of the episodes start with the team gathered within two cells of one road cell and with full batteries; half start as in training, spread out and with random battery levels. One target stands farther from every capturer than twice that capturer's initial charge, and for the capturer nearest to it the way around the obstacles is at least four cells longer than the straight distance. The episode lasts 128 steps {E}. The rule acts greedily: one lattice, no noise, no search. Success is a capture.</li>
</ul>
<h3>What the benchmark must guarantee, and how each property is certified</h3>
''' + table(["Property", "Certificate, computed outside the learner", "Control, run on every reported model"], [
        ["A recharge is required", "For every capturer, the target starts more than twice that capturer's initial charge away. A capturer that is not refilled acts for as many steps as its charge, and the gap closes by at most two cells per step. This is arithmetic on Manhattan distance, as in the 8×8 task, and it holds for any start layout",
         "Rechargers removed, or left in the team with refills switched off: exactly 0. Rechargers frozen where they start: at most 5%"],
        ["Obstacles must be flown around", f"For the capturer nearest to the target, the obstacle-respecting distance exceeds the Manhattan distance by 4 or more. Needs a shortest-path computation in the generator {Q}",
         "Obstacle channel zeroed in the rule's input: at most half the headline rate"],
        ["Coordination is required", "With a sight of two cells, a lone drone next to the target always leaves it a safe option, so no single drone can capture where every free cell has two free neighbours. Hand proof; to be confirmed by enumeration on small grids",
         "One capturer alone: exactly 0"],
        ["Capture is possible", f"A scripted team that always sees the target captures in at least 1 of 16 attempts on the instance, and in at least 90% of attempts over the whole set. A hand-written heuristic, used only to admit instances {Q}",
         "The script's rate is published as a reference. Our rate is also reported on the unfiltered set, with the share of instances that were rejected"],
        ["Scouting has value", f"A second scripted team, which acts on what the team has sensed instead of on the true position, loses at least 10 points when the observers' sensing is withheld. Otherwise the radii are recalibrated before any training {Q}",
         "Observers removed; observers flying with their sensing withheld"],
    ]) + f'''
<p>Controls that apply to a team are first run with the scripted teams, to show that the benchmark has the property, and later with every model that is reported. The benchmark is frozen at the end of phase 1. If a control later fails to behave as stated, the learner is never adjusted to pass it. A change of environment values then needs the PI's approval and becomes a new pre-registered version on which earlier results are reported again.</p>
<h3>Pre-registration</h3>
<p>Before the first training run on a benchmark version, the following go into git: the parameter file, the generator version, the seed blocks, the control thresholds, and hashes of the validation and test sets. Test: 500 new maps with 4 episodes each. Validation: 200 maps with 2 episodes each. One checkpoint per training seed, chosen on validation, is evaluated once on the test set. Intervals come from a bootstrap that resamples training seeds first and maps second.</p>
<p><strong>The bar</strong> {Q}. Proposed: the lower end of the 95% interval is at least 80% for the mean over five training seeds, each trained from scratch, on the 2,000 test episodes from new maps of the training families, with the greedy readout. The held-out map family and the hand-drawn maps are reported beside it against a secondary bar of 60%. Phase 5 uses the point estimate on validation as a development gate; only phase 6 tests the bar.</p>
''')

    # ------------------------------------------------------------------ 4
    S.append(h2(4, "contract", "The bitter-lesson contract") + f'''
<p>Sutton's bitter lesson says that methods which scale with computation, namely search and learning, end up ahead of methods that build in what their designers know. Version 1 turned that into five rules. Version 2 keeps them with one exception, one addition and one rule that had to be loosened.</p>
<ol class="rules">
  <li><strong>No domain concept in the learner.</strong> Distance, meeting, escort, pincer, sweep, low battery, sighting and role are not named anywhere in it. <em>Check</em> a text search of the learner's source. <em>Exception</em> each drone's battery level is an input {D}.</li>
  <li><strong>Every ability comes from search or learning,</strong> and has a compute dial (§11).</li>
  <li><strong>More compute never hurts.</strong> More iterations, a longer window or more clones must not lower success. <em>Check</em> §10. One of the two 8×8 checkpoints fails this today {M}.</li>
  <li><strong>No shaped reward, no demonstration, no staged curriculum, no solution heuristic.</strong> Tasks are drawn from one declared distribution. Selection uses the return and nothing else. <em>Loosened</em> version 1 also excluded every exploration prior and every curriculum. Version 2 admits two things that wording excluded, and the PI has accepted both: exploration noise held over several steps, and a training mixture over grid sizes and team sizes (§12, §15).</li>
  <li><strong>Every prior that remains is written down</strong> with the test that could remove it (§12).</li>
  <li><strong>The benchmark may know what the learner may not</strong> {Q}. Shortest paths, connectivity checks and the scripted teams exist only in the generator and the evaluator. Nothing they compute reaches the learner as an input, as a filter on training tasks or as a training signal. The learner's code imports none of theirs.</li>
</ol>
<p>What the learner is told about a drone: its cell, its kind as one of three fixed labels, its battery level, and a code that changes every episode. The kind label is inherited from the 8×8 design, where each type had its own channel. What a kind can do is learned.</p>
<p>What the learner takes from the simulator. For acting: the next observation, the reward, and the cell an action leads to. For search and credit, in training only {D}: saving and restoring a state, drawing what is hidden afresh, replaying a leg with some drones acting at random, and running a leg with one drone's sensing withheld.</p>
''')

    # ------------------------------------------------------------------ 5
    S.append(h2(5, "lattice", "The lattice") + f'''
<p>The lattice has time as one of its three axes. It holds a window of slices around the present: 3 behind it, the present, and 16 ahead. Slices behind hold facts. Slices ahead hold the dream, in which each drone has a tube that marks where it will be. The present is the front between the two.</p>
<figure>
  <div class="scroll diagram">{b.lattice_figure()}</div>
  <figcaption>The window on a 4×4 excerpt, drawn with a longer past than the design uses. Behind the front the input holds where the drones really were. Ahead of it the rule has grown the tubes of a capturer and a recharger onto the same cells: they meet at t+1 and reach the target at t+3.</figcaption>
</figure>
''' + table(["Channel group", "Count", "Content", "Non-zero in"], [
        ["Map (input)", "2", "Obstacle, road", "Every slice"],
        ["Sensed (input)", "2", "This cell was inside a sensing radius; the target was found here", "Past and present"],
        ["Kind and code (input)", "3 × 16", "For each kind, the sum of the codes of the drones of that kind on this cell", "Past and present"],
        ["Battery (input)", "16", "The sum of battery level × code over the drones on this cell", "Past and present"],
        ["State", "32", "Free for the rule to use; kept from step to step", "Everywhere"],
        ["Tube (output)", "16", "A vector per voxel. Drone i's tube is its inner product with code i", "Read in future slices"],
        ["Total", "116", "68 input, 32 state, 16 output", ""],
    ], num=(1,)) + f'''
<p><strong>What the input holds.</strong> Clamping writes the input channels of the present slice after every step: which cells were sensed, the target if it was sensed, and each drone's kind, code and battery. When the window slides, that slice becomes a past slice and keeps what was written. Future slices hold the map and nothing else. At reset the past slices hold the map only. A battery level is given in steps divided by 24, so that the largest capacity reads 1; a recharger always reads 1. A drone with an empty battery stays in the input with level zero. Its tube is still read, but the simulator turns every move of an empty drone into a stay. The lattice is not told that a capture happened or that a new target exists; it sees only what the sensors report.</p>
<p><strong>Codes.</strong> At the start of an episode every drone receives its own one-hot vector of length 16, drawn at random. Its tube is the inner product of the output vector with that code, which is one of the sixteen output channels. Because the assignment is redrawn every episode, no channel can come to mean a fixed drone or role. A rule built from rectified units can gate one channel without touching the others. Codes drawn as random rotations are the alternative to test {U}.</p>
''' + b.pre([
        ("c  = conv 3×3×3 ( X )", "once per step: the inputs do not change between iterations"),
        ("h  = relu( conv 3×3×3 ( S ) + c )", "zero padding; the only place neighbours are seen"),
        ("S  ← bound( S + W3 · relu( W2 · h ) )", "bound: divide by max(1, root-mean-square over the voxel's channels)"),
        ("Y  = Wout · S", "the 16 tube channels, one voxel at a time"),
    ], 43) + f'''
<p>All weights are shared across voxels, slices and iterations. There are no coordinate channels, no pooling, no attention and no iteration counter. With 68 input channels, 32 state channels and 128 hidden units the rule has about 370,000 weights. The kernel sees one slice back and one ahead, so information flows from the present towards the future and from a future meeting back to the present. The state is kept in full precision and only the convolutions run in a reduced format.</p>
<p><strong>Plans longer than the window.</strong> A tube shows the next 16 steps of a drone's route. Whatever lies further ahead is carried by the state, which spreads across the map at one cell per iteration: 48 cells at reset, then 4 per step, while drones move 1. The labels a tube is trained on are the cells the drone really occupied over the next 16 steps of a kept leg, whether or not the capture lies inside the window.</p>
''')

    # ------------------------------------------------------------------ 6
    S.append(h2(6, "acting", "Acting") + b.pre([
        "reset",
        ("    S ← 0;  X ← map in every slice", ""),
        "    give every drone a code",
        "    clamp the first observation into the present slice",
        "    run the rule K0 times",
        "",
        "every step t",
        "    for every drone i, all at once",
        ("        for each action a:  u[a] ← sim.peek(i, a)", "the cell the action leads to"),
        ("        score[a] ← ⟨ Y[present + 1][u[a]], code_i ⟩", ""),
        ("        a_i ← argmax score", "when searching: a sampled action (§7)"),
        "    observation, reward ← sim.step(a_1 … a_n)",
        ("    slide S and X one slice towards the past", "the new last slice: S = 0, X = map"),
        ("    clamp the observation into the present slice", ""),
        "    run the rule k_s times",
    ], 62) + f'''
<ul>
  <li>Drones are steered by tubes, which are positions in space-time, and never by a field of arrows. A drone moves to the reachable cell where its own tube is strongest in the next slice.</li>
  <li>There is no order among drones. One step costs k_s iterations of the rule, whatever the size of the team.</li>
  <li>The lattice keeps its state. A plan made at one step is still there at the next unless new facts contradict it {U}.</li>
  <li>When evaluating there is one lattice, no noise and no search.</li>
  <li><strong>Commit groups, a dial that starts at one.</strong> To commit is to clamp a drone's chosen next cell into the next slice before other drones choose. With two groups, a random half of the drones chooses and commits, the rule runs once, and then the other half chooses. This is version 1's mechanism at a fraction of its cost, and it is switched on only if two drones of one kind keep ending up on one strand.</li>
</ul>
''')

    S.append("@@SEARCH@@")
    S.append("@@CREDIT@@")
    S.append("@@LEARNING@@")
    S.append("@@GATES@@")
    S.append("@@COMPUTE@@")
    S.append("@@PRIORS@@")
    S.append("@@ROADMAP@@")
    S.append("@@CLUSTER@@")
    S.append("@@DECISIONS@@")
    return S
