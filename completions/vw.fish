complete --command vw --long-option ring_size --description 'ring_size: size of example ring' --no-files --require-parameter
complete --command vw --long-option strict_parse --description 'strict_parse: throw on malformed examples' --no-files
complete --command vw --long-option learning_rate --short-option l --description 'learning_rate: Set learning rate' --no-files --require-parameter
complete --command vw --long-option power_t --description 'power_t: t power value' --no-files --require-parameter
complete --command vw --long-option decay_learning_rate --description 'decay_learning_rate: Set Decay factor for learning_rate between passes' --no-files --require-parameter
complete --command vw --long-option initial_t --description 'initial_t: initial t value' --no-files --require-parameter
complete --command vw --long-option feature_mask --description 'feature_mask: Use existing regressor to determine which parameters may be updated.  If no initial_regressor given, also used for initial weights.' --no-files --require-parameter
complete --command vw --long-option initial_regressor --short-option i --description 'initial_regressor: Initial regressor(s)' --force-files --require-parameter
complete --command vw --long-option initial_weight --description 'initial_weight: Set all weights to an initial value of arg.' --no-files --require-parameter
complete --command vw --long-option random_weights --description 'random_weights: make initial weights random' --no-files
complete --command vw --long-option normal_weights --description 'normal_weights: make initial weights normal' --no-files
complete --command vw --long-option truncated_normal_weights --description 'truncated_normal_weights: make initial weights truncated normal' --no-files
complete --command vw --long-option sparse_weights --description 'sparse_weights: Use a sparse datastructure for weights' --no-files
complete --command vw --long-option input_feature_regularizer --description 'input_feature_regularizer: Per feature regularization input file' --force-files --require-parameter
complete --command vw --long-option span_server --description 'span_server: Location of server for setting up spanning tree' --no-files --require-parameter
complete --command vw --long-option unique_id --description 'unique_id: unique id used for cluster parallel jobs' --no-files --require-parameter
complete --command vw --long-option total --description 'total: total number of nodes used in cluster parallel job' --no-files --require-parameter
complete --command vw --long-option node --description 'node: node number in cluster parallel job' --no-files --require-parameter
complete --command vw --long-option span_server_port --description 'span_server_port: Port of the server for setting up spanning tree' --no-files --require-parameter
complete --command vw --long-option version --description 'version: Version information' --no-files
complete --command vw --long-option audit --short-option a --description 'audit: print weights of features' --no-files
complete --command vw --long-option progress --short-option P --description 'progress: Progress update frequency. int: additive, float: multiplicative' --no-files --require-parameter
complete --command vw --long-option quiet --description 'quiet: Don\'t output disgnostics and progress updates' --no-files
complete --command vw --long-option limit_output --description 'limit_output: Avoid chatty output. Limit total printed lines.' --no-files --require-parameter
complete --command vw --long-option dry_run --description 'dry_run: Parse arguments and print corresponding metadata. Will not execute driver.' --no-files
complete --command vw --long-option help --short-option h --description 'help: More information on vowpal wabbit can be found here https://vowpalwabbit.org.' --no-files
complete --command vw --long-option random_seed --description 'random_seed: seed random number generator' --no-files --require-parameter
complete --command vw --long-option hash --description 'hash: how to hash the features. Available options: strings, all' --no-files --require-parameter
complete --command vw --long-option hash_seed --description 'hash_seed: seed for hash function' --no-files --require-parameter
complete --command vw --long-option ignore --description 'ignore: ignore namespaces beginning with character <arg>' --no-files --require-parameter
complete --command vw --long-option ignore_linear --description 'ignore_linear: ignore namespaces beginning with character <arg> for linear terms only' --no-files --require-parameter
complete --command vw --long-option keep --description 'keep: keep namespaces beginning with character <arg>' --no-files --require-parameter
complete --command vw --long-option redefine --description 'redefine: redefine namespaces beginning with characters of std::string S as namespace N. <arg> shall be in form \'N:=S\' where := is operator. Empty N or S are treated as default namespace. Use \':\' as a wildcard in S.' --no-files --require-parameter
complete --command vw --long-option bit_precision --short-option b --description 'bit_precision: number of bits in the feature table' --no-files --require-parameter
complete --command vw --long-option noconstant --description 'noconstant: Don\'t add a constant feature' --no-files
complete --command vw --long-option constant --short-option C --description 'constant: Set initial value of constant' --no-files --require-parameter
complete --command vw --long-option ngram --description 'ngram: Generate N grams. To generate N grams for a single namespace \'foo\', arg should be fN.' --no-files --require-parameter
complete --command vw --long-option skips --description 'skips: Generate skips in N grams. This in conjunction with the ngram tag can be used to generate generalized n-skip-k-gram. To generate n-skips for a single namespace \'foo\', arg should be fN.' --no-files --require-parameter
complete --command vw --long-option feature_limit --description 'feature_limit: limit to N features. To apply to a single namespace \'foo\', arg should be fN' --no-files --require-parameter
complete --command vw --long-option affix --description 'affix: generate prefixes/suffixes of features; argument \'+2a,-3b,+1\' means generate 2-char prefixes for namespace a, 3-char suffixes for b and 1 char prefixes for default namespace' --no-files --require-parameter
complete --command vw --long-option spelling --description 'spelling: compute spelling features for a give namespace (use \'_\' for default namespace)' --no-files --require-parameter
complete --command vw --long-option dictionary --description 'dictionary: read a dictionary for additional features (arg either \'x:file\' or just \'file\')' --no-files --require-parameter
complete --command vw --long-option dictionary_path --description 'dictionary_path: look in this directory for dictionaries; defaults to current directory or env{PATH}' --no-files --require-parameter
complete --command vw --long-option interactions --description 'interactions: Create feature interactions of any level between namespaces.' --no-files --require-parameter
complete --command vw --long-option permutations --description 'permutations: Use permutations instead of combinations for feature interactions of same namespace.' --no-files
complete --command vw --long-option leave_duplicate_interactions --description 'leave_duplicate_interactions: Don\'t remove interactions with duplicate combinations of namespaces. For ex. this is a duplicate: \'-q ab -q ba\' and a lot more in \'-q ::\'.' --no-files
complete --command vw --long-option quadratic --short-option q --description 'quadratic: Create and use quadratic features' --no-files --require-parameter
complete --command vw --long-option q: --description 'q:: DEPRECATED \':\' corresponds to a wildcard for all printable characters' --no-files --require-parameter
complete --command vw --long-option cubic --description 'cubic: Create and use cubic features' --no-files --require-parameter
complete --command vw --long-option testonly --short-option t --description 'testonly: Ignore label information and just test' --no-files
complete --command vw --long-option holdout_off --description 'holdout_off: no holdout data in multiple passes' --no-files
complete --command vw --long-option holdout_period --description 'holdout_period: holdout period for test only' --no-files --require-parameter
complete --command vw --long-option holdout_after --description 'holdout_after: holdout after n training examples, default off (disables holdout_period)' --no-files --require-parameter
complete --command vw --long-option early_terminate --description 'early_terminate: Specify the number of passes tolerated when holdout loss doesn\'t decrease before early termination' --no-files --require-parameter
complete --command vw --long-option passes --description 'passes: Number of Training Passes' --no-files --require-parameter
complete --command vw --long-option initial_pass_length --description 'initial_pass_length: initial number of examples per pass' --no-files --require-parameter
complete --command vw --long-option examples --description 'examples: number of examples to parse' --no-files --require-parameter
complete --command vw --long-option min_prediction --description 'min_prediction: Smallest prediction to output' --no-files --require-parameter
complete --command vw --long-option max_prediction --description 'max_prediction: Largest prediction to output' --no-files --require-parameter
complete --command vw --long-option sort_features --description 'sort_features: turn this on to disregard order in which features have been defined. This will lead to smaller cache sizes' --no-files
complete --command vw --long-option loss_function --description 'loss_function: Specify the loss function to be used, uses squared by default. Currently available ones are squared, classic, hinge, logistic, quantile and poisson.' --no-files --require-parameter
complete --command vw --long-option quantile_tau --description 'quantile_tau: Parameter \tau associated with Quantile loss. Defaults to 0.5' --no-files --require-parameter
complete --command vw --long-option l1 --description 'l1: l_1 lambda' --no-files --require-parameter
complete --command vw --long-option l2 --description 'l2: l_2 lambda' --no-files --require-parameter
complete --command vw --long-option no_bias_regularization --description 'no_bias_regularization: no bias in regularization' --no-files
complete --command vw --long-option named_labels --description 'named_labels: use names for labels (multiclass, etc.) rather than integers, argument specified all possible labels, comma-sep, eg "--named_labels Noun,Verb,Adj,Punc"' --no-files --require-parameter
complete --command vw --long-option final_regressor --short-option f --description 'final_regressor: Final regressor' --force-files --require-parameter
complete --command vw --long-option readable_model --description 'readable_model: Output human-readable final regressor with numeric features' --no-files --require-parameter
complete --command vw --long-option invert_hash --description 'invert_hash: Output human-readable final regressor with feature names.  Computationally expensive.' --no-files --require-parameter
complete --command vw --long-option save_resume --description 'save_resume: save extra state so learning can be resumed later with new data' --no-files
complete --command vw --long-option preserve_performance_counters --description 'preserve_performance_counters: reset performance counters when warmstarting' --no-files
complete --command vw --long-option save_per_pass --description 'save_per_pass: Save the model after every pass over data' --no-files
complete --command vw --long-option output_feature_regularizer_binary --description 'output_feature_regularizer_binary: Per feature regularization output file' --force-files --require-parameter
complete --command vw --long-option output_feature_regularizer_text --description 'output_feature_regularizer_text: Per feature regularization output file, in text' --force-files --require-parameter
complete --command vw --long-option id --description 'id: User supplied ID embedded into the final regressor' --no-files --require-parameter
complete --command vw --long-option predictions --short-option p --description 'predictions: File to output predictions to' --no-files --require-parameter
complete --command vw --long-option raw_predictions --short-option r --description 'raw_predictions: File to output unnormalized predictions to' --no-files --require-parameter
complete --command vw --long-option extra_metrics --description 'extra_metrics: Specify filename to write metrics to. Note: There is no fixed schema.' --no-files --require-parameter
complete --command vw --long-option audit_regressor --description 'audit_regressor: stores feature names and their regressor values. Same dataset must be used for both regressor training and this mode.' --force-files --require-parameter
complete --command vw --long-option search --description 'search: Use learning to search, argument=maximum action id or 0 for LDF' --no-files --require-parameter
complete --command vw --long-option search_task --description 'search_task: the search task (use "--search_task list" to get a list of available tasks)' --no-files --require-parameter
complete --command vw --long-option search_metatask --description 'search_metatask: the search metatask (use "--search_metatask list" to get a list of available metatasks)' --no-files --require-parameter
complete --command vw --long-option search_interpolation --description 'search_interpolation: at what level should interpolation happen? [*data|policy]' --no-files --require-parameter
complete --command vw --long-option search_rollout --description 'search_rollout: how should rollouts be executed?           [policy|oracle|*mix_per_state|mix_per_roll|none]' --no-files --require-parameter
complete --command vw --long-option search_rollin --description 'search_rollin: how should past trajectories be generated? [policy|oracle|*mix_per_state|mix_per_roll]' --no-files --require-parameter
complete --command vw --long-option search_passes_per_policy --description 'search_passes_per_policy: number of passes per policy (only valid for search_interpolation=policy)' --no-files --require-parameter
complete --command vw --long-option search_beta --description 'search_beta: interpolation rate for policies (only valid for search_interpolation=policy)' --no-files --require-parameter
complete --command vw --long-option search_alpha --description 'search_alpha: annealed beta = 1-(1-alpha)^t (only valid for search_interpolation=data)' --no-files --require-parameter
complete --command vw --long-option search_total_nb_policies --description 'search_total_nb_policies: if we are going to train the policies through multiple separate calls to vw, we need to specify this parameter and tell vw how many policies are eventually going to be trained' --no-files --require-parameter
complete --command vw --long-option search_trained_nb_policies --description 'search_trained_nb_policies: the number of trained policies in a file' --no-files --require-parameter
complete --command vw --long-option search_allowed_transitions --description 'search_allowed_transitions: read file of allowed transitions [def: all transitions are allowed]' --no-files --require-parameter
complete --command vw --long-option search_subsample_time --description 'search_subsample_time: instead of training at all timesteps, use a subset. if value in (0,1), train on a random v%. if v>=1, train on precisely v steps per example, if v<=-1, use active learning' --no-files --require-parameter
complete --command vw --long-option search_neighbor_features --description 'search_neighbor_features: copy features from neighboring lines. argument looks like: \'-1:a,+2\' meaning copy previous line namespace a and next next line from namespace _unnamed_, where \',\' separates them' --no-files --require-parameter
complete --command vw --long-option search_rollout_num_steps --description 'search_rollout_num_steps: how many calls of "loss" before we stop really predicting on rollouts and switch to oracle (default means "infinite")' --no-files --require-parameter
complete --command vw --long-option search_history_length --description 'search_history_length: some tasks allow you to specify how much history their depend on; specify that here' --no-files --require-parameter
complete --command vw --long-option search_no_caching --description 'search_no_caching: turn off the built-in caching ability (makes things slower, but technically more safe)' --no-files
complete --command vw --long-option search_xv --description 'search_xv: train two separate policies, alternating prediction/learning' --no-files
complete --command vw --long-option search_perturb_oracle --description 'search_perturb_oracle: perturb the oracle on rollin with this probability' --no-files --require-parameter
complete --command vw --long-option search_linear_ordering --description 'search_linear_ordering: insist on generating examples in linear order (def: hoopla permutation)' --no-files
complete --command vw --long-option search_active_verify --description 'search_active_verify: verify that active learning is doing the right thing (arg = multiplier, should be = cost_range * range_c)' --no-files --require-parameter
complete --command vw --long-option search_save_every_k_runs --description 'search_save_every_k_runs: save model every k runs' --no-files --require-parameter
complete --command vw --long-option replay_c --description 'replay_c: use experience replay at a specified level [b=classification/regression, m=multiclass, c=cost sensitive] with specified buffer size' --no-files --require-parameter
complete --command vw --long-option replay_c_count --description 'replay_c_count: how many times (in expectation) should each example be played (default: 1 = permuting)' --no-files --require-parameter
complete --command vw --long-option ot --description 'ot: Offset tree with <k> labels' --no-files --require-parameter
complete --command vw --long-option cb_to_cbadf --description 'cb_to_cbadf: Maps cb_adf to cb. Disable with cb_force_legacy.' --no-files --require-parameter
complete --command vw --long-option cb --description 'cb: Maps cb_adf to cb. Disable with cb_force_legacy.' --no-files --require-parameter
complete --command vw --long-option cb_explore --description 'cb_explore: Translate cb explore to cb_explore_adf. Disable with cb_force_legacy.' --no-files --require-parameter
complete --command vw --long-option cbify --description 'cbify: Translate cbify to cb_adf. Disable with cb_force_legacy.' --no-files --require-parameter
complete --command vw --long-option cb_type --description 'cb_type: contextual bandit method to use in {}' --no-files --require-parameter
complete --command vw --long-option cb_force_legacy --description 'cb_force_legacy: Default to non-adf cb implementation (cb_algs)' --no-files
complete --command vw --long-option cbify_ldf --description 'cbify_ldf: Convert csoaa_ldf into a contextual bandit problem' --no-files
complete --command vw --long-option loss0 --description 'loss0: loss for correct label' --no-files --require-parameter
complete --command vw --long-option loss1 --description 'loss1: loss for incorrect label' --no-files --require-parameter
complete --command vw --long-option cbify_cs --description 'cbify_cs: Consume cost-sensitive classification examples instead of multiclass' --no-files
complete --command vw --long-option cbify_reg --description 'cbify_reg: Consume regression examples instead of multiclass and cost sensitive' --no-files
complete --command vw --long-option cats --description 'cats: Continuous action tree with smoothing' --no-files --require-parameter
complete --command vw --long-option cb_discrete --description 'cb_discrete: Discretizes continuous space and adds cb_explore as option' --no-files
complete --command vw --long-option min_value --description 'min_value: Minimum continuous value' --no-files --require-parameter
complete --command vw --long-option max_value --description 'max_value: Maximum continuous value' --no-files --require-parameter
complete --command vw --long-option loss_option --description 'loss_option: loss options for regression - 0:squared, 1:absolute, 2:0/1' --no-files --require-parameter
complete --command vw --long-option loss_report --description 'loss_report: loss report option - 0:normalized, 1:denormalized' --no-files --require-parameter
complete --command vw --long-option loss_01_ratio --description 'loss_01_ratio: ratio of zero loss for 0/1 loss' --no-files --require-parameter
complete --command vw --long-option bandwidth --description 'bandwidth: Bandwidth (radius) of randomization around discrete actions in terms of continuous range. By default will be set to half of the continuous action unit-range resulting in smoothing that stays inside the action space unit-range:unit_range = (max_value - min_value)/num-of-actionsdefault bandwidth = unit_range / 2.0' --no-files --require-parameter
complete --command vw --long-option sample_pdf --description 'sample_pdf: Sample a pdf and pick a continuous valued action' --no-files
complete --command vw --long-option cats_pdf --description 'cats_pdf: number of tree labels <k> for cats_pdf' --no-files --require-parameter
complete --command vw --long-option cb_explore_pdf --description 'cb_explore_pdf: Sample a pdf and pick a continuous valued action' --no-files
complete --command vw --long-option epsilon --description 'epsilon: epsilon-greedy exploration' --no-files --require-parameter
complete --command vw --long-option first_only --description 'first_only: Use user provided first action or user provided pdf or uniform random' --no-files
complete --command vw --long-option pmf_to_pdf --description 'pmf_to_pdf: number of discrete actions <k> for pmf_to_pdf' --no-files --require-parameter
complete --command vw --long-option get_pmf --description 'get_pmf: Convert a single multiclass prediction to a pmf' --no-files
complete --command vw --long-option warm_cb --description 'warm_cb: Convert multiclass on <k> classes into a contextual bandit problem' --no-files --require-parameter
complete --command vw --long-option warm_cb_cs --description 'warm_cb_cs: consume cost-sensitive classification examples instead of multiclass' --no-files
complete --command vw --long-option warm_start --description 'warm_start: number of training examples for warm start phase' --no-files --require-parameter
complete --command vw --long-option interaction --description 'interaction: number of examples for the interactive contextual bandit learning phase' --no-files --require-parameter
complete --command vw --long-option warm_start_update --description 'warm_start_update: indicator of warm start updates' --no-files
complete --command vw --long-option interaction_update --description 'interaction_update: indicator of interaction updates' --no-files
complete --command vw --long-option corrupt_type_warm_start --description 'corrupt_type_warm_start: type of label corruption in the warm start phase (1: uniformly at random, 2: circular, 3: replacing with overwriting label)' --no-files --require-parameter
complete --command vw --long-option corrupt_prob_warm_start --description 'corrupt_prob_warm_start: probability of label corruption in the warm start phase' --no-files --require-parameter
complete --command vw --long-option choices_lambda --description 'choices_lambda: the number of candidate lambdas to aggregate (lambda is the importance weight parameter between the two sources)' --no-files --require-parameter
complete --command vw --long-option lambda_scheme --description 'lambda_scheme: The scheme for generating candidate lambda set (1: center lambda=0.5, 2: center lambda=0.5, min lambda=0, max lambda=1, 3: center lambda=epsilon/(1+epsilon), 4: center lambda=epsilon/(1+epsilon), min lambda=0, max lambda=1); the rest of candidate lambda values are generated using a doubling scheme' --no-files --require-parameter
complete --command vw --long-option overwrite_label --description 'overwrite_label: the label used by type 3 corruptions (overwriting)' --no-files --require-parameter
complete --command vw --long-option sim_bandit --description 'sim_bandit: simulate contextual bandit updates on warm start examples' --no-files
complete --command vw --long-option slates --description 'slates: EXPERIMENTAL' --no-files
complete --command vw --long-option ccb_explore_adf --description 'ccb_explore_adf: EXPERIMENTAL: Do Conditional Contextual Bandit learning with multiline action dependent features.' --no-files
complete --command vw --long-option all_slots_loss --description 'all_slots_loss: Report average loss from all slots' --no-files
complete --command vw --long-option explore_eval --description 'explore_eval: Evaluate explore_eval adf policies' --no-files
complete --command vw --long-option multiplier --description 'multiplier: Multiplier used to make all rejection sample probabilities <= 1' --no-files --require-parameter
complete --command vw --long-option cb_sample --description 'cb_sample: Sample from CB pdf and swap top action.' --no-files
complete --command vw --long-option cb_dro --description 'cb_dro: Use DRO for cb learning' --no-files
complete --command vw --long-option cb_dro_alpha --description 'cb_dro_alpha: Confidence level for cb dro' --no-files --require-parameter
complete --command vw --long-option cb_dro_tau --description 'cb_dro_tau: Time constant for count decay for cb dro' --no-files --require-parameter
complete --command vw --long-option cb_dro_wmax --description 'cb_dro_wmax: maximum importance weight for cb_dro' --no-files --require-parameter
complete --command vw --long-option cb_explore_adf --description 'cb_explore_adf: Online explore-exploit for a contextual bandit problem with multiline action dependent features' --no-files
complete --command vw --long-option bag --description 'bag: bagging-based exploration' --no-files --require-parameter
complete --command vw --long-option greedify --description 'greedify: always update first policy once in bagging' --no-files
complete --command vw --long-option cover --description 'cover: Online cover based exploration' --no-files --require-parameter
complete --command vw --long-option psi --description 'psi: disagreement parameter for cover' --no-files --require-parameter
complete --command vw --long-option nounif --description 'nounif: do not explore uniformly on zero-probability actions in cover' --no-files
complete --command vw --long-option first --description 'first: tau-first exploration' --no-files --require-parameter
complete --command vw --long-option synthcover --description 'synthcover: use synthetic cover exploration' --no-files
complete --command vw --long-option synthcoverpsi --description 'synthcoverpsi: exploration reward bonus' --no-files --require-parameter
complete --command vw --long-option synthcoversize --description 'synthcoversize: number of policies in cover' --no-files --require-parameter
complete --command vw --long-option squarecb --description 'squarecb: SquareCB exploration' --no-files
complete --command vw --long-option gamma_scale --description 'gamma_scale: Sets SquareCB greediness parameter to gamma=[gamma_scale]*[num examples]^1/2' --no-files --require-parameter
complete --command vw --long-option gamma_exponent --description 'gamma_exponent: Exponent on [num examples] in SquareCB greediness parameter gamma.' --no-files --require-parameter
complete --command vw --long-option elim --description 'elim: Only perform SquareCB exploration over plausible actions (computed via RegCB strategy)' --no-files
complete --command vw --long-option mellowness --description 'mellowness: Mellowness parameter c_0 for computing plausible action set. Only used with --elim' --no-files --require-parameter
complete --command vw --long-option cb_min_cost --description 'cb_min_cost: Lower bound on cost. Only used with --elim' --no-files --require-parameter
complete --command vw --long-option cb_max_cost --description 'cb_max_cost: Upper bound on cost. Only used with --elim' --no-files --require-parameter
complete --command vw --long-option regcb --description 'regcb: RegCB-elim exploration' --no-files
complete --command vw --long-option regcbopt --description 'regcbopt: RegCB optimistic exploration' --no-files
complete --command vw --long-option rnd --description 'rnd: rnd based exploration' --no-files --require-parameter
complete --command vw --long-option rnd_alpha --description 'rnd_alpha: ci width for rnd (bigger => more exploration on repeating features)' --no-files --require-parameter
complete --command vw --long-option rnd_invlambda --description 'rnd_invlambda: covariance regularization strength rnd (bigger => more exploration on new features)' --no-files --require-parameter
complete --command vw --long-option softmax --description 'softmax: softmax exploration' --no-files
complete --command vw --long-option lambda --description 'lambda: parameter for softmax' --no-files --require-parameter
complete --command vw --long-option cats_tree --description 'cats_tree: CATS Tree with <k> labels' --no-files --require-parameter
complete --command vw --long-option tree_bandwidth --description 'tree_bandwidth: tree bandwidth for continuous actions in terms of #actions' --no-files --require-parameter
complete --command vw --long-option link --description 'link: Specify the link function: identity, logistic, glf1 or poisson' --no-files --require-parameter
complete --command vw --long-option multiworld_test --description 'multiworld_test: Evaluate features as a policies' --no-files --require-parameter
complete --command vw --long-option learn --description 'learn: Do Contextual Bandit learning on <n> classes.' --no-files --require-parameter
complete --command vw --long-option exclude_eval --description 'exclude_eval: Discard mwt policy features before learning' --no-files
complete --command vw --long-option cb_adf --description 'cb_adf: Do Contextual Bandit learning with multiline action dependent features.' --no-files
complete --command vw --long-option rank_all --description 'rank_all: Return actions sorted by score order' --no-files
complete --command vw --long-option no_predict --description 'no_predict: Do not do a prediction when training' --no-files
complete --command vw --long-option clip_p --description 'clip_p: Clipping probability in importance weight. Default: 0.f (no clipping).' --no-files --require-parameter
complete --command vw --long-option eval --description 'eval: Evaluate a policy rather than optimizing.' --no-files
complete --command vw --long-option csoaa_ldf --description 'csoaa_ldf: Use one-against-all multiclass learning with label dependent features.' --no-files --require-parameter
complete --command vw --long-option ldf_override --description 'ldf_override: Override singleline or multiline from csoaa_ldf or wap_ldf, eg if stored in file' --no-files --require-parameter
complete --command vw --long-option csoaa_rank --description 'csoaa_rank: Return actions sorted by score order' --no-files
complete --command vw --long-option probabilities --description 'probabilities: predict probabilites of all classes' --no-files
complete --command vw --long-option wap_ldf --description 'wap_ldf: Use weighted all-pairs multiclass learning with label dependent features.  Specify singleline or multiline.' --no-files --require-parameter
complete --command vw --long-option interact --description 'interact: Put weights on feature products from namespaces <n1> and <n2>' --no-files --require-parameter
complete --command vw --long-option csoaa --description 'csoaa: One-against-all multiclass with <k> costs' --no-files --require-parameter
complete --command vw --long-option cs_active --description 'cs_active: Cost-sensitive active learning with <k> costs' --no-files --require-parameter
complete --command vw --long-option simulation --description 'simulation: cost-sensitive active learning simulation mode' --no-files
complete --command vw --long-option baseline --description 'baseline: cost-sensitive active learning baseline' --no-files
complete --command vw --long-option domination --description 'domination: cost-sensitive active learning use domination. Default 1' --no-files --require-parameter
complete --command vw --long-option range_c --description 'range_c: parameter controlling the threshold for per-label cost uncertainty. Default 0.5.' --no-files --require-parameter
complete --command vw --long-option max_labels --description 'max_labels: maximum number of label queries.' --no-files --require-parameter
complete --command vw --long-option min_labels --description 'min_labels: minimum number of label queries.' --no-files --require-parameter
complete --command vw --long-option cost_max --description 'cost_max: cost upper bound. Default 1.' --no-files --require-parameter
complete --command vw --long-option cost_min --description 'cost_min: cost lower bound. Default 0.' --no-files --require-parameter
complete --command vw --long-option csa_debug --description 'csa_debug: print debug stuff for cs_active' --no-files
complete --command vw --long-option plt --description 'plt: Probabilistic Label Tree with <k> labels' --no-files --require-parameter
complete --command vw --long-option kary_tree --description 'kary_tree: use <k>-ary tree' --no-files --require-parameter
complete --command vw --long-option threshold --description 'threshold: predict labels with conditional marginal probability greater than <thr> threshold' --no-files --require-parameter
complete --command vw --long-option top_k --description 'top_k: predict top-<k> labels instead of labels above threshold' --no-files --require-parameter
complete --command vw --long-option multilabel_oaa --description 'multilabel_oaa: One-against-all multilabel with <k> labels' --no-files --require-parameter
complete --command vw --long-option classweight --description 'classweight: importance weight multiplier for class' --no-files --require-parameter
complete --command vw --long-option memory_tree --description 'memory_tree: Make a memory tree with at most <n> nodes' --no-files --require-parameter
complete --command vw --long-option max_number_of_labels --description 'max_number_of_labels: max number of unique label' --no-files --require-parameter
complete --command vw --long-option leaf_example_multiplier --description 'leaf_example_multiplier: multiplier on examples per leaf (default = log nodes)' --no-files --require-parameter
complete --command vw --long-option alpha --description 'alpha: Alpha' --no-files --require-parameter
complete --command vw --long-option dream_repeats --description 'dream_repeats: number of dream operations per example (default = 1)' --no-files --require-parameter
complete --command vw --long-option top_K --description 'top_K: top K prediction error (default 1)' --no-files --require-parameter
complete --command vw --long-option learn_at_leaf --description 'learn_at_leaf: Enable learning at leaf' --no-files
complete --command vw --long-option oas --description 'oas: use oas at the leaf' --no-files
complete --command vw --long-option dream_at_update --description 'dream_at_update: turn on dream operations at reward based update as well' --no-files --require-parameter
complete --command vw --long-option online --description 'online: turn on dream operations at reward based update as well' --no-files
complete --command vw --long-option recall_tree --description 'recall_tree: Use online tree for multiclass' --no-files --require-parameter
complete --command vw --long-option max_candidates --description 'max_candidates: maximum number of labels per leaf in the tree' --no-files --require-parameter
complete --command vw --long-option bern_hyper --description 'bern_hyper: recall tree depth penalty' --no-files --require-parameter
complete --command vw --long-option max_depth --description 'max_depth: maximum depth of the tree, default log_2 (#classes)' --no-files --require-parameter
complete --command vw --long-option node_only --description 'node_only: only use node features, not full path features' --no-files
complete --command vw --long-option randomized_routing --description 'randomized_routing: randomized routing' --no-files
complete --command vw --long-option log_multi --description 'log_multi: Use online tree for multiclass' --no-files --require-parameter
complete --command vw --long-option no_progress --description 'no_progress: disable progressive validation' --no-files
complete --command vw --long-option swap_resistance --description 'swap_resistance: higher = more resistance to swap, default=4' --no-files --require-parameter
complete --command vw --long-option ect --description 'ect: Error correcting tournament with <k> labels' --no-files --require-parameter
complete --command vw --long-option error --description 'error: errors allowed by ECT' --no-files --require-parameter
complete --command vw --long-option boosting --description 'boosting: Online boosting with <N> weak learners' --no-files --require-parameter
complete --command vw --long-option gamma --description 'gamma: weak learner\'s edge (=0.1), used only by online BBM' --no-files --require-parameter
complete --command vw --long-option alg --description 'alg: specify the boosting algorithm: BBM (default), logistic (AdaBoost.OL.W), adaptive (AdaBoost.OL)' --no-files --require-parameter
complete --command vw --long-option oaa --description 'oaa: One-against-all multiclass with <k> labels' --no-files --require-parameter
complete --command vw --long-option oaa_subsample --description 'oaa_subsample: subsample this number of negative examples when learning' --no-files --require-parameter
complete --command vw --long-option scores --description 'scores: output raw scores per class' --no-files
complete --command vw --long-option top --description 'top: top k recommendation' --no-files --require-parameter
complete --command vw --long-option replay_m --description 'replay_m: use experience replay at a specified level [b=classification/regression, m=multiclass, c=cost sensitive] with specified buffer size' --no-files --require-parameter
complete --command vw --long-option replay_m_count --description 'replay_m_count: how many times (in expectation) should each example be played (default: 1 = permuting)' --no-files --require-parameter
complete --command vw --long-option binary --description 'binary: report loss as binary classification on -1,1' --no-files
complete --command vw --long-option bootstrap --description 'bootstrap: k-way bootstrap by online importance resampling' --no-files --require-parameter
complete --command vw --long-option bs_type --description 'bs_type: prediction type {mean,vote}' --no-files --require-parameter
complete --command vw --long-option cbzo --description 'cbzo: Solve 1-slot Continuous Action Contextual Bandit using Zeroth-Order Optimization' --no-files
complete --command vw --long-option policy --description 'policy: Policy/Model to Learn' --no-files --require-parameter
complete --command vw --long-option radius --description 'radius: Exploration Radius' --no-files --require-parameter
complete --command vw --long-option stage_poly --description 'stage_poly: use stagewise polynomial feature learning' --no-files
complete --command vw --long-option sched_exponent --description 'sched_exponent: exponent controlling quantity of included features' --no-files --require-parameter
complete --command vw --long-option batch_sz --description 'batch_sz: multiplier on batch size before including more features' --no-files --require-parameter
complete --command vw --long-option batch_sz_no_doubling --description 'batch_sz_no_doubling: batch_sz does not double' --no-files
complete --command vw --long-option lrqfa --description 'lrqfa: use low rank quadratic features with field aware weights' --no-files --require-parameter
complete --command vw --long-option lrq --description 'lrq: use low rank quadratic features' --no-files --require-parameter
complete --command vw --long-option lrqdropout --description 'lrqdropout: use dropout training for low rank quadratic features' --no-files
complete --command vw --long-option autolink --description 'autolink: create link function with polynomial d' --no-files --require-parameter
complete --command vw --long-option marginal --description 'marginal: substitute marginal label estimates for ids' --no-files --require-parameter
complete --command vw --long-option initial_denominator --description 'initial_denominator: initial denominator' --no-files --require-parameter
complete --command vw --long-option initial_numerator --description 'initial_numerator: initial numerator' --no-files --require-parameter
complete --command vw --long-option compete --description 'compete: enable competition with marginal features' --no-files
complete --command vw --long-option update_before_learn --description 'update_before_learn: update marginal values before learning' --no-files
complete --command vw --long-option unweighted_marginals --description 'unweighted_marginals: ignore importance weights when computing marginals' --no-files
complete --command vw --long-option decay --description 'decay: decay multiplier per event (1e-3 for example)' --no-files --require-parameter
complete --command vw --long-option nn --description 'nn: Sigmoidal feedforward network with <k> hidden units' --no-files --require-parameter
complete --command vw --long-option inpass --description 'inpass: Train or test sigmoidal feedforward network with input passthrough.' --no-files
complete --command vw --long-option multitask --description 'multitask: Share hidden layer across all reduced tasks.' --no-files
complete --command vw --long-option dropout --description 'dropout: Train or test sigmoidal feedforward network using dropout.' --no-files
complete --command vw --long-option meanfield --description 'meanfield: Train or test sigmoidal feedforward network using mean field.' --no-files
complete --command vw --long-option confidence --description 'confidence: Get confidence for binary predictions' --no-files
complete --command vw --long-option confidence_after_training --description 'confidence_after_training: Confidence after training' --no-files
complete --command vw --long-option active_cover --description 'active_cover: enable active learning with cover' --no-files
complete --command vw --long-option beta_scale --description 'beta_scale: active learning variance upper bound parameter beta_scale. Default std::sqrt(10).' --no-files --require-parameter
complete --command vw --long-option oracular --description 'oracular: Use Oracular-CAL style query or not. Default false.' --no-files
complete --command vw --long-option active --description 'active: enable active learning' --no-files
complete --command vw --long-option replay_b --description 'replay_b: use experience replay at a specified level [b=classification/regression, m=multiclass, c=cost sensitive] with specified buffer size' --no-files --require-parameter
complete --command vw --long-option replay_b_count --description 'replay_b_count: how many times (in expectation) should each example be played (default: 1 = permuting)' --no-files --require-parameter
complete --command vw --long-option lr_multiplier --description 'lr_multiplier: learning rate multiplier for baseline model' --no-files --require-parameter
complete --command vw --long-option global_only --description 'global_only: use separate example with only global constant for baseline predictions' --no-files
complete --command vw --long-option check_enabled --description 'check_enabled: only use baseline when the example contains enabled flag' --no-files
complete --command vw --long-option new_mf --description 'new_mf: rank for reduction-based matrix factorization' --no-files --require-parameter
complete --command vw --long-option OjaNewton --description 'OjaNewton: Online Newton with Oja\'s Sketch' --no-files
complete --command vw --long-option sketch_size --description 'sketch_size: size of sketch' --no-files --require-parameter
complete --command vw --long-option epoch_size --description 'epoch_size: size of epoch' --no-files --require-parameter
complete --command vw --long-option alpha_inverse --description 'alpha_inverse: one over alpha, similar to learning rate' --no-files --require-parameter
complete --command vw --long-option learning_rate_cnt --description 'learning_rate_cnt: constant for the learning rate 1/t' --no-files --require-parameter
complete --command vw --long-option normalize --description 'normalize: normalize the features or not' --no-files --require-parameter
complete --command vw --long-option random_init --description 'random_init: randomize initialization of Oja or not' --no-files --require-parameter
complete --command vw --long-option conjugate_gradient --description 'conjugate_gradient: use conjugate gradient based optimization' --no-files
complete --command vw --long-option bfgs --description 'bfgs: use conjugate gradient based optimization' --no-files
complete --command vw --long-option hessian_on --description 'hessian_on: use second derivative in line search' --no-files
complete --command vw --long-option mem --description 'mem: memory in bfgs' --no-files --require-parameter
complete --command vw --long-option termination --description 'termination: Termination threshold' --no-files --require-parameter
complete --command vw --long-option lda --description 'lda: Run lda with <int> topics' --no-files --require-parameter
complete --command vw --long-option lda_alpha --description 'lda_alpha: Prior on sparsity of per-document topic weights' --no-files --require-parameter
complete --command vw --long-option lda_rho --description 'lda_rho: Prior on sparsity of topic distributions' --no-files --require-parameter
complete --command vw --long-option lda_D --description 'lda_D: Number of documents' --no-files --require-parameter
complete --command vw --long-option lda_epsilon --description 'lda_epsilon: Loop convergence threshold' --no-files --require-parameter
complete --command vw --long-option minibatch --description 'minibatch: Minibatch size, for LDA' --no-files --require-parameter
complete --command vw --long-option math-mode --description 'math-mode: Math mode: simd, accuracy, fast-approx' --no-files --require-parameter
complete --command vw --long-option metrics --description 'metrics: Compute metrics' --no-files
complete --command vw --long-option noop --description 'noop: do no learning' --no-files
complete --command vw --long-option print --description 'print: print examples' --no-files
complete --command vw --long-option rank --description 'rank: rank for matrix factorization.' --no-files --require-parameter
complete --command vw --long-option sendto --description 'sendto: send examples to <host>' --no-files --require-parameter
complete --command vw --long-option svrg --description 'svrg: Streaming Stochastic Variance Reduced Gradient' --no-files
complete --command vw --long-option stage_size --description 'stage_size: Number of passes per SVRG stage' --no-files --require-parameter
complete --command vw --long-option ftrl --description 'ftrl: FTRL: Follow the Proximal Regularized Leader' --no-files
complete --command vw --long-option coin --description 'coin: Coin betting optimizer' --no-files
complete --command vw --long-option pistol --description 'pistol: PiSTOL: Parameter-free STOchastic Learning' --no-files
complete --command vw --long-option ftrl_alpha --description 'ftrl_alpha: Learning rate for FTRL optimization' --no-files --require-parameter
complete --command vw --long-option ftrl_beta --description 'ftrl_beta: Learning rate for FTRL optimization' --no-files --require-parameter
complete --command vw --long-option ksvm --description 'ksvm: kernel svm' --no-files
complete --command vw --long-option reprocess --description 'reprocess: number of reprocess steps for LASVM' --no-files --require-parameter
complete --command vw --long-option pool_greedy --description 'pool_greedy: use greedy selection on mini pools' --no-files
complete --command vw --long-option para_active --description 'para_active: do parallel active learning' --no-files
complete --command vw --long-option pool_size --description 'pool_size: size of pools for active learning' --no-files --require-parameter
complete --command vw --long-option subsample --description 'subsample: number of items to subsample from the pool' --no-files --require-parameter
complete --command vw --long-option kernel --description 'kernel: type of kernel (rbf or linear (default))' --no-files --require-parameter
complete --command vw --long-option degree --description 'degree: degree of poly kernel' --no-files --require-parameter
complete --command vw --long-option sgd --description 'sgd: use regular stochastic gradient descent update.' --no-files
complete --command vw --long-option adaptive --description 'adaptive: use adaptive, individual learning rates.' --no-files
complete --command vw --long-option adax --description 'adax: use adaptive learning rates with x^2 instead of g^2x^2' --no-files
complete --command vw --long-option invariant --description 'invariant: use safe/importance aware updates.' --no-files
complete --command vw --long-option normalized --description 'normalized: use per feature normalized updates' --no-files
complete --command vw --long-option sparse_l2 --description 'sparse_l2: use per feature normalized updates' --no-files --require-parameter
complete --command vw --long-option l1_state --description 'l1_state: use per feature normalized updates' --no-files --require-parameter
complete --command vw --long-option l2_state --description 'l2_state: use per feature normalized updates' --no-files --require-parameter
complete --command vw --long-option data --short-option d --description 'data: Example set' --force-files --require-parameter
complete --command vw --long-option daemon --description 'daemon: persistent daemon mode on port 26542' --no-files
complete --command vw --long-option foreground --description 'foreground: in persistent daemon mode, do not run in the background' --no-files
complete --command vw --long-option port --description 'port: port to listen on; use 0 to pick unused port' --no-files --require-parameter
complete --command vw --long-option num_children --description 'num_children: number of children for persistent daemon mode' --no-files --require-parameter
complete --command vw --long-option pid_file --description 'pid_file: Write pid file in persistent daemon mode' --no-files --require-parameter
complete --command vw --long-option port_file --description 'port_file: Write port used in persistent daemon mode' --no-files --require-parameter
complete --command vw --long-option cache --short-option c --description 'cache: Use a cache.  The default is <data>.cache' --no-files
complete --command vw --long-option cache_file --description 'cache_file: The location(s) of cache_file.' --force-files --require-parameter
complete --command vw --long-option json --description 'json: Enable JSON parsing.' --no-files
complete --command vw --long-option dsjson --description 'dsjson: Enable Decision Service JSON parsing.' --no-files
complete --command vw --long-option kill_cache --short-option k --description 'kill_cache: do not reuse existing cache: create a new one always' --no-files
complete --command vw --long-option compressed --description 'compressed: use gzip format whenever possible. If a cache file is being created, this option creates a compressed cache file. A mixture of raw-text & compressed inputs are supported with autodetection.' --no-files
complete --command vw --long-option no_stdin --description 'no_stdin: do not default to reading from stdin' --no-files
complete --command vw --long-option no_daemon --description 'no_daemon: Force a loaded daemon or active learning model to accept local input instead of starting in daemon mode' --no-files
complete --command vw --long-option chain_hash --description 'chain_hash: Enable chain hash in JSON for feature name and string feature value. e.g. {\'A\': {\'B\': \'C\'}} is hashed as A^B^C. Note: this will become the default in a future version, so enabling this option will migrate you to the new behavior and silence the warning.' --no-files
complete --command vw --long-option flatbuffer --description 'flatbuffer: data file will be interpreted as a flatbuffer file' --no-files
