    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False

    # unique identifier
    uid = uuid.uuid4().hex

    # initialize seeds
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # def load_datasets(args):
    d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
    n_inputs = d_tr[0][1].size(1)
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max())
        n_outputs = max(n_outputs, d_te[i][2].max())
    x_tr, x_te, n_inputs, n_outputs, n_tasks =  d_tr, d_te, n_inputs, n_outputs.item() + 1, len(d_tr)


    continuum = Continuum(x_tr, args)
    #
    #
    # class Continuum:
    #     def __init__(self, data, args):
    #         self.data = x_tr
    #         self.batch_size = args.batch_size
    #         n_tasks = len(x_tr)
    #         task_permutation = range(n_tasks)
    #
    #         if args.shuffle_tasks == 'yes':
    #             task_permutation = torch.randperm(n_tasks).tolist()
    #
    #         sample_permutations = []
    #
    #         for t in range(n_tasks):
    #             N = x_tr[t][1].size(0)
    #             if args.samples_per_task <= 0:
    #                 n = N
    #             else:
    #                 n = min(args.samples_per_task, N)
    #
    #             p = torch.randperm(N)[0:n]
    #             sample_permutations.append(p)
    #
    #         self.permutation = []
    #
    #         for t in range(n_tasks):
    #             task_t = task_permutation[t]
    #             for _ in range(args.n_epochs):
    #                 task_p = [[task_t, i] for i in sample_permutations[task_t]]
    #                 random.shuffle(task_p)
    #                 self.permutation += task_p
    #
    #         self.length = len(self.permutation)
    #         self.current = 0
    #         # import pdb; pdb.set_trace()
    #     def __iter__(self):
    #         return self
    #
    #     def next(self):
    #         return self.__next__()
    #
    #     def __next__(self):
    #         if self.current >= self.length:
    #             raise StopIteration
    #         else:
    #             ti = self.permutation[self.current][0]
    #             j = []
    #             i = 0
    #             while (((self.current + i) < self.length) and
    #                    (self.permutation[self.current + i][0] == ti) and
    #                    (i < self.batch_size)):
    #                 j.append(self.permutation[self.current + i][1])
    #                 i += 1
    #             self.current += i
    #             j = torch.LongTensor(j)
    #             return self.x_tr[ti][1][j], ti, self.x_tr[ti][2][j]
        # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(n_inputs, n_outputs, n_tasks, args)
    device = torch.device("cuda" if args.cuda else "cpu")
    model.to(device)
    # run model on continuum
    # life_experience(
    #     model, continuum, x_te, args)
    #
    # def life_experience(model, continuum, x_te, args):
    result_a = []
    result_t = []

    current_task = 0
    time_start = time.time()

    for (i, (x, t, y)) in enumerate(continuum):
        # import pdb; pdb.set_trace()
        if(((i % args.log_every) == 0) or (t != current_task)):
            result_a.append(eval_tasks(model, x_te, args))
            result_t.append(current_task)
            current_task = t

        v_x = x.view(x.size(0), -1)
        v_y = y.long()
        device = torch.device("cuda" if args.cuda else "cpu")
        # if args.cuda:
        #     v_x = v_x.cuda()
        #     v_y = v_y.cuda()
        v_x = v_x.to(device)
        v_y = v_y.to(device)
        model.train()
        # model.observe(Variable(v_x), t, Variable(v_y))
        model.observe(v_x, t, v_y)
        # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    result_a.append(eval_tasks(model, x_te, args))
    result_t.append(current_task)

    time_end = time.time()
    time_spent = time_end - time_start
    # import pdb; pdb.set_trace()
    result_t, result_a, spent_time = torch.Tensor(result_t), torch.Tensor(result_a), time_spent

    # # prepare saving path and file name
    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)
    #
    # fname = args.model + '_' + args.data_file + '_'
    # fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # fname += '_' + uid
    # fname = os.path.join(args.save_path, fname)
    #
    # # save confusion matrix and print one line of stats
    # stats = confusion_matrix(result_t, result_a, fname + '.txt')
    # one_liner = str(vars(args)) + ' # '
    # one_liner += ' '.join(["%.3f" % stat for stat in stats])
    # print(fname + ': ' + one_liner + ' # ' + str(spent_time))
    #
    # # save all results in binary file
    # torch.save((result_t, result_a, model.state_dict(),
    #             stats, one_liner, args), fname + '.pt')
