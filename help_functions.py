def get_texts(fname, minwords = 100):
    texts = []
    with fname.open('r', encoding='utf-8') as f:
        curr = ['', 0]
        for line in f:
            if line == '\n': # giannis
                continue
            l = len(line.split(' '))
            if curr[1] + l > minwords:
                texts.append(curr[0])
                curr = [line, l]
            else:
                curr[0] += '\n' + line
                curr[1] += l
    if curr[0] != '':
        texts.append(curr[0])
    return np.array(texts)


def get_sentences(fname):
    texts = []
    with fname.open('r', encoding='utf-8') as f:
        for line in f:
            if line == '\n': continue
            line = re.sub(r"\[[0-9]+\]", "", line) # replace "[<int>]" by "" (common in dfw_lobster)
            sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', line.strip())
            texts.extend(sents)
    return texts


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ').replace('.',' .').replace('?',' ?').replace('!',' !').replace('’'," '")
    return re1.sub(' ', html.unescape(x))


def preprocess(all_texts):
    col_names = ['text']
    df = pd.DataFrame({'text':all_texts}, columns=col_names)
    re1 = re.compile(r'  +')

    texts = df['text'].astype(str)
    texts = list(texts.apply(fixup).values)
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok


def get_vocab(min_freq, max_freq, keep_general_vocab, sample_general_vocab, savefile):
    freq = Counter(p for o in tok_trn for p in o)
    
    # get domain vocab
    itos = [o for o,c in freq.most_common(max_freq) if c>min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')

    # get general vocab
    itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
    if sample_general_vocab != -1:
        itos2 = random.sample(itos2, sample_general_vocab) # giannis
    stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})

    if keep_general_vocab:
        itos = list(set(itos).union(itos2))
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

    with open(savefile, 'wb') as f:
        pickle.dump(itos, f)

    print("itos: {}".format(len(itos)))
    print("itos2: {}".format(len(itos2)))
    print("itos2 - itos: {}".format(len(set(itos2)-set(itos))))
    print("itos - itos2: {}".format(len(set(itos)-set(itos2))))
    print("itos & itos2: {}".format(len(set(itos).union(set(itos2)))))
    #unseen = set(itos) - set(itos2)
    #print("Domain words not seen previously ({} in total): {}".format(len(unseen), unseen))
    print("Vocab size = {}".format(len(itos)))
    return itos, stoi, itos2, stoi2


def get_pretrained_lm(itos, stoi2):
    em_sz,nh,nl = 400,1150,3

    PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'
    print('loading pre-trained model from {} ...'.format(PRE_LM_PATH))
    wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)

    enc_wgts = to_np(wgts['0.encoder.weight'])
    row_m = enc_wgts.mean(0)

    new_w = np.zeros((vs, em_sz), dtype=np.float32)
    for i,w in enumerate(itos):                     # for word in imbd vocab
        r = stoi2[w]                                # get the int in the pretrained vocab
        new_w[i] = enc_wgts[r] if r>=0 else row_m   # add weight if in vocab, else add mean weight

    wgts['0.encoder.weight'] = T(new_w)
    wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
    wgts['1.decoder.weight'] = T(np.copy(new_w))

    bptt=70
    bs=52
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
    val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
    md = LanguageModelData(LM_PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7

    learner = md.get_model(opt_fn, em_sz, nh, nl, 
        dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

    learner.model.load_state_dict(wgts)
    print('done')
    return learner