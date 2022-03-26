import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
import torch.nn.functional as F
from tqdm import tqdm

# 参数
#data_folder = 'G:/第五章系统/data/generateData'  # 包含create_input_File.py保存的数据文件的文件夹
data_folder = './data/generateData'  # 包含create_input_File.py保存的数据文件的文件夹
data_name = '_' + str(1) + '_cap_per_img_'

# checkpoint = 'G:/第五章系统/data/生成模型/CIE-X-Linear.pth.tar'  # model checkpoint

#word_map_file = r'G:\第五章系统\data\generateData\WORDMAP.json'
word_map_file = r'.\data\generateData\WORDMAP.json'
device = torch.device("cpu" )
cudnn.benchmark = True

# Load model
# checkpoint = torch.load(checkpoint,map_location="cpu")
# # encoder = checkpoint['encoder']
# encoder = encoder.to(device)
# encoder.eval()
# # mlc = checkpoint['mlc']
# mlc = mlc.to(device)
# mlc.eval()

# decoder = checkpoint['decoder']
# decoder = decoder.to(device)
# decoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r', encoding='utf-8') as j:
    word_map = json.load(j)

# 词映射的逆映射   之前是词-数字   现在数字-词
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)



def evaluate(loadpath,flag=2,model_name='CIE-X-Linear.pth.tar'):
    """
    Evaluation

    :param beam_size: 生成句子用beam search 评估
    :return: BLEU-4 score
    """
    print(loadpath,flag)
    if(flag==1):
        data_folder = './data/generateData'  # 包含create_input_File.py保存的数据文件的文件夹
        data_name = '_' + str(1) + '_cap_per_img_'

        device = torch.device("cpu")
        cudnn.benchmark = True
        checkpoint='./data/generatereport/'+model_name
        word_map_file = r'.data\generateData\WORDMAP.json'
        checkpoint = torch.load(checkpoint, map_location="cpu")
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()
        mlc = checkpoint['mlc']
        mlc = mlc.to(device)
        mlc.eval()

        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        # Load word map (word2ix)
        with open(word_map_file, 'r', encoding='utf-8') as j:
            word_map = json.load(j)

        # 词映射的逆映射   之前是词-数字   现在数字-词
        rev_word_map = {v: k for k, v in word_map.items()}
        vocab_size = len(word_map)
    else:
        data_folder = './data/generateData'  # 包含create_input_File.py保存的数据文件的文件夹
        data_name = '_' + str(1) + '_cap_per_img_'

        device = torch.device("cpu")
        cudnn.benchmark = True
        # checkpoint='G:/第五章系统/data/生成模型/胎儿心脏超声/'+model_name
        # word_map_file = r'G:\第五章系统\data\generateData\WORDMAP_heart.json'
        checkpoint='./data/generatereport/'+model_name
        word_map_file = r'.\data\generateData\WORDMAP_heart.json'
        checkpoint = torch.load(checkpoint, map_location="cpu")
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()
        mlc = checkpoint['mlc']
        mlc = mlc.to(device)
        mlc.eval()

        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        # Load word map (word2ix)
        with open(word_map_file, 'r', encoding='utf-8') as j:
            word_map = json.load(j)

        # 词映射的逆映射   之前是词-数字   现在数字-词
        rev_word_map = {v: k for k, v in word_map.items()}
        vocab_size = len(word_map)
    # DataLoader
    print("果然")
    loader = torch.utils.data.DataLoader(
        CaptionDataset( data_name, 'TEST',loadpath),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # TODO: 批量beam search
    # 因此，不要使用大于1的批处理大小-重要！

    # 为每个图像存储引用（真实字幕）和假设（预测）的列表
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # 对于每个图像
    for i, (all_feats) in enumerate(
        tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(1))):

        all_feats = all_feats.to(device)

        all_feats,_ = encoder(all_feats)  # (1,7,7,2048)
        b = all_feats.size(0)
        e = all_feats.size(3)
        mean_feats = torch.mean(all_feats.view(b, -1, e), dim=1)
        mean_feats = mean_feats.to(device)
        pre_tag, semantic_features = mlc(mean_feats)
        semantic_features = semantic_features.to(device)

        k = 1

        # Move to GPU device, if available


        # Encode
        encoder_out = all_feats  # (1, enc_image_size, enc_image_size, encoder_dim)  （1,14,14,2048）
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)


        # 我们将把这个问题当作批量大小为k的问题
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
        detect_feats = encoder_out.squeeze(1)  # (k,2048)
        
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)


        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()



        # 展平图像
        all_feats = all_feats.view(1, -1, encoder_dim)  # (K, num_pixels, encoder_dim) 1？
        # CNN_feats= decoder.dropout(decoder.feat_embed(all_feats))
        CNN_feats = decoder.feat_embed(all_feats)  #  (1,49,1024)
        Q = torch.mean(CNN_feats, dim=1)  # (1,1024)

        Q1, K, V = decoder.X_Linear(Q, CNN_feats, CNN_feats)
        #Q2, K, V = decoder.X_Linear(Q1,K, V)
        #att_feats = decoder.W_G(torch.cat((Q, Q1,Q2), dim=1))  # (1,1024)
        att_feats = decoder.W_G(torch.cat((Q, Q1), dim=1))  # (1,1024)

        att_feats = att_feats.expand(k, K.size(2))
        global_feats = V . expand(k, K.size(1) ,K.size(2))  # (1,7*7,1024)

        
        
         # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(all_feats)
        h=h.expand(k,h.size(1))
        c=c.expand(k,h.size(1))
        

        mean_feat = Q
        ctx_X_Linear = torch.zeros_like(h)  #(1, 1024)
        ctx_X_Linear=ctx_X_Linear.expand(k,  1024)

        semantic_features=semantic_features.squeeze(1)
        semantic_features=semantic_features.expand(k, semantic_features.size(1))
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        
        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            Input = torch.cat((att_feats + ctx_X_Linear, embeddings),dim=1)
            h, c = decoder.decode_step(Input, (h, c))  # (s, decoder_dim)
            v_mao_d, _, _ = decoder.X_Linear(h, global_feats, global_feats)
            ctx_X_Linear = F.glu(torch.cat((v_mao_d, h), dim=1))  # (s, decoder_dim)
            scores = decoder.fc(ctx_X_Linear)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            
            #print("------------------",prev_word_inds,"---------",next_word_inds,"---------",top_k_words,"---",vocab_size,"++++++",top_k_words // vocab_size)
            prev_word_inds=prev_word_inds.long()
            next_word_inds = next_word_inds.long()
            
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            att_feats=att_feats[prev_word_inds[incomplete_inds]]
            semantic_features=semantic_features[prev_word_inds[incomplete_inds]]
            detect_feats = detect_feats[prev_word_inds[incomplete_inds]]
            global_feats=global_feats[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            ctx_X_Linear=ctx_X_Linear[prev_word_inds[incomplete_inds]]
            
            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        pre_caps = [rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        pre_caps_num=[w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        str_res = ''
        seq_res=''
        res=""
        for i in range(len(pre_caps)):
            res+=pre_caps[i]
            str_res += pre_caps[i] + ' '
            seq_res+= str(pre_caps_num[i])+' '
        a=loadpath.split("/")

        # with open('G:/第五章系统/data/generateData/Result/report.txt', 'a+', encoding='utf-8') as f:
        #         f.write(str(a[-1]) + '\t' + res.strip(' ') + '\n')
        with open('./data/generateData/Result/report.txt', 'a+', encoding='utf-8') as f:
                f.write(str(a[-1]) + '\t' + res.strip(' ') + '\n')

        im = Image.open(loadpath)

        print(a,a[-1])
        # im.save("G:/第五章系统/data/generateData/Result/"+a[-1])
        im.save("./data/generateData/Result/"+a[-1])

        print(res)
    return res,a[-1]


if __name__ == '__main__':

    #print(evaluate("G:/第五章系统/image/四腔心样例.png",2,'CIE-X-Linear.pth.tar'))
    print(evaluate("./image/四腔心样例.png",2,'CIE-X-Linear.pth.tar'))
