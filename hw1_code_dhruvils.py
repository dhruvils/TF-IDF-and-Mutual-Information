#! python
import os
from math import log, sqrt
from nltk import regexp_tokenize
from collections import Counter
from operator import itemgetter

# global variables:

dhruvils_doc_freq = dict()
dhruvils_idf_freq = dict()
dhruvils_corpus_root = '/home1/c/cis530/hw1/data/corpus'
dhruvils_brown_cluster_path = '/home1/c/cis530/hw1/data/brownwc.txt'

def get_all_files(dir):
    relativeFileList = []
    for dirpath, dirs, files in os.walk(dir):
        relativeFileList += [ (dirpath.replace(dir, '')) + ('' if dirpath == dir else '/') + filename for filename in files]
    return relativeFileList

def load_file_tokens(filepath):
    user_file = open(filepath, 'r')
    file_contents = user_file.read()
    user_file.close()
    return regexp_tokenize(file_contents.lower(), '[a-z0-9]\w+')

def load_collection_tokens(dir):
    relativeFileList = []
    tokens = []

    for dirpath, dirs, files in os.walk(dir):
        relativeFileList += [dirpath + '/' + filename for filename in files]

    for filepath in relativeFileList:
        tokens += load_file_tokens(filepath)

    return tokens

def get_tf(itemlist):
    term_freq = Counter(itemlist)
    most_freq = max(term_freq.values())
    
    for key in term_freq:
        term_freq[key] /= float(most_freq)

    return term_freq

def get_tf_top(itemlist, k):
    term_freq = get_tf(itemlist)
    sorted_tf = sorted(term_freq, key=term_freq.get, reverse=True)
    return sorted_tf[:k if k < len(sorted_tf) else len(sorted_tf)]

def calc_doc_freq(itemlist):  # used to calculate the doc freqs of all words once from @itemlist (2D list containing words within docs)
    global dhruvils_doc_freq
    doc_set_list = []

    for doc in itemlist:
        doc_set_list.append(set(doc))

    for doc in doc_set_list:
        for word in doc:
            if word in dhruvils_doc_freq:
                dhruvils_doc_freq[word] += 1
            else:
                dhruvils_doc_freq[word] = 1

    return dhruvils_doc_freq

def get_doc_freq(word):  # used by idf to find the doc_freq of @word from pre calculated @doc_freq
    return dhruvils_doc_freq[word]

def get_doc_wise_tokens(dir):  # used to generate the 2D itemlist for get_idf()
    relativeFileList = []
    tokens = []

    for dirpath, dirs, files in os.walk(dir):
        relativeFileList += [ dirpath + "/" + filename for filename in files]

    for filepath in relativeFileList:
        tokens.append(load_file_tokens(filepath))

    return tokens

def get_idf(itemlist):
    global dhruvils_idf_freq
    num_doc = len(itemlist)
    for doc in itemlist:
        for word in doc:
            if word not in dhruvils_idf_freq:
                doc_freq = get_doc_freq(word)
                dhruvils_idf_freq[word] = log(num_doc / float(doc_freq))
    dhruvils_idf_freq['<UNK>'] = log(float(num_doc))
    return dhruvils_idf_freq

def get_tfidf_top(tf_values_dict, idf_values_dict, k):
    tfidf_dict = dict()
    for token in tf_values_dict:
        if token in idf_values_dict:
            tfidf_dict[token] = tf_values_dict[token] * idf_values_dict[token]
        else:
            tfidf_dict[token] = tf_values_dict[token] * idf_values_dict['<UNK>']
    
    sorted_term_list = sorted(tfidf_dict, key=tfidf_dict.get, reverse=True)
    return sorted_term_list[:k if k < len(sorted_term_list) else len(sorted_term_list)]

def get_mi_top(bg_terms, topic_terms, k):
    mi_dict = dict()
    #total_bg_terms = len(bg_terms)
    count_bg_terms = Counter(bg_terms)
    count_bg_terms = {k: v for k, v in count_bg_terms.items() if v >= 5}
    total_bg_terms = 0
    for term in count_bg_terms:
        total_bg_terms += count_bg_terms[term]
    total_topic_terms = len(topic_terms)
    count_topic_terms = Counter(topic_terms)
    #count_topic_terms = {k: v for k, v in count_topic_terms.items() if v >= 5}
    
    for word in count_topic_terms:
        if word in count_bg_terms:
            con_prob_of_word = count_topic_terms[word] / float(total_topic_terms)
            gen_prob_of_word = count_bg_terms[word] / float(total_bg_terms)
            mi = log(con_prob_of_word / gen_prob_of_word)
            mi_dict[word] = mi if mi > 0 else 0

    sorted_mi_list = sorted(mi_dict, key=mi_dict.get, reverse = True)
    return sorted_mi_list[:k if k < len(sorted_mi_list) else len(sorted_mi_list)]

def get_mi_dict(bg_terms, topic_terms):
    mi_dict = dict()
    #total_bg_terms = len(bg_terms)
    count_bg_terms = Counter(bg_terms)
    count_bg_terms = {k: v for k, v in count_bg_terms.items() if v >= 5}
    total_bg_terms = 0
    for term in count_bg_terms:
        total_bg_terms += count_bg_terms[term]
    total_topic_terms = len(topic_terms)
    count_topic_terms = Counter(topic_terms)
    #count_topic_terms = {k: v for k, v in count_topic_terms.items() if v >= 5}
    
    for word in count_topic_terms:
        if word in count_bg_terms:
            con_prob_of_word = count_topic_terms[word] / float(total_topic_terms)
            gen_prob_of_word = count_bg_terms[word] / float(total_bg_terms)
            mi = log(con_prob_of_word / gen_prob_of_word)
            mi_dict[word] = mi if mi > 0 else 0

    return mi_dict

def write_mi_weights(directory, outfilename):
    topic_terms = load_collection_tokens(directory)
    bg_terms = load_collection_tokens(dhruvils_corpus_root)

    mi_dict = get_mi_dict(bg_terms, topic_terms)
    mi_dict = {k: v for k, v in mi_dict.items() if v > 0}
    #mi_dict = sorted(mi_dict, key = mi_dict.get, reverse = True)
    output_file = open(outfilename, 'w')

    for word in mi_dict:
        output_file.write(word + '\t' +str(mi_dict[word]) +'\n')

    output_file.close()

def get_precision(L1, L2):
    return len(set.intersection(set(L1), L2)) / float(len(L1))

def get_recall(L1, L2):
    return len(set.intersection(set(L1), L2)) / float(len(L2))

def get_fmeasure(L1, L2):
    precision = get_precision(L1, L2)
    recall = get_recall(L1, L2)
    return (2 * precision * recall) / (float(precision + recall))

def write_comp_results(mi_list, tf_list, tfidf_list):
    mi_precision = get_precision(mi_list, tfidf_list)
    mi_recall = get_recall(mi_list, tfidf_list)

    tf_precision = get_precision(tf_list, tfidf_list)
    tf_recall = get_recall(tf_list, tfidf_list)

    outputfile = open('result.txt', 'w')
    outputfile.write(str(mi_precision) +', ' +str(mi_recall) +'\n')
    outputfile.write(str(tf_precision) +', ' +str(tf_recall) +'\n')
    outputfile.close()

def read_brown_cluster():
    brown_dict = dict()
    
    with open(dhruvils_brown_cluster_path) as brownfile:
        for line in brownfile:
            (value, key, discard) = line.split()
            brown_dict[key] = value

    return brown_dict

def load_file_clusters(filepath, bc_dict):
    cluster_id_list = []
    file_tokens = load_file_tokens(filepath)
    
    for token in file_tokens:
        if token in bc_dict:
            cluster_id_list.append(bc_dict[token])

    return cluster_id_list

def load_collection_clusters(directory, bc_dict):
    cluster_id_list = []
    dir_tokens = load_collection_tokens(directory)

    for token in dir_tokens:
        if token in bc_dict:
            cluster_id_list.append(bc_dict[token])

    return cluster_id_list

def load_docwise_collection_clusters(directory, bc_dict):
    cluster_id_list = []
    dir_tokens = get_doc_wise_tokens(directory)
    
    for doc in dir_tokens:
        doc_cluster_id_list = []
        for token in doc:
            if token in bc_dict:
                doc_cluster_id_list.append(bc_dict[token])
        cluster_id_list.append(doc_cluster_id_list)

    return cluster_id_list

def get_idf_clusters(bc_dict):
    itemlist = load_docwise_collection_clusters('/home1/c/cis530/hw1/data/all_d\
ata/', bc_dict)
    calc_doc_freq(itemlist)
    return get_idf(itemlist)

def get_tfidf_clusters(tf_values_dict, idf_values_dict):
    tfidf_dict = dict()
    for token in tf_values_dict:
        if token in idf_values_dict:
            tfidf_dict[token] = tf_values_dict[token] * idf_values_dict[token]
        else:
            tfidf_dict[token] = tf_values_dict[token] * idf_values_dict['<UNK>']
    return tfidf_dict

def get_tf_clusters(itemlist):
    return get_tf(itemlist)

def write_tfidf_weights(directory, outfilename, bc_dict):
    idf_values_dict = get_idf_clusters(bc_dict)
    itemlist = load_collection_clusters(directory, bc_dict)
    tf_values_dict = get_tf_clusters(itemlist)

    tfidf_dict = get_tfidf_clusters(tf_values_dict, idf_values_dict)
    
    outfile = open(outfilename, 'w')
    for key in tfidf_dict:
        outfile.write(str(key) +'\t' +str(tfidf_dict[key]) +'\n')
    outfile.close()

def create_feature_space(clusters_list):
    vec = 0
    feature_space = dict()

    for term in clusters_list:
        if term not in feature_space:
            feature_space[term] = vec
            vec += 1

    return feature_space

def vectorize(feature_space, cluster_id_list):
    vec = []
    
    for key in feature_space:
        if key in cluster_id_list:
            vec.append(1)
        else:
            vec.append(0)
    
    return vec

def cosine_similarity(x, y):
    n = len(x)
    count = 0
    num = 0
    card_x = 0
    card_y = 0
    
    for count in range (0, n):
        num += x[count] * y[count]
        card_x += x[count]**2
        card_y += y[count]**2

    if card_x == 0 or card_y == 0:
        cos_sim = 0
    else:
        cos_sim = float(num) / (sqrt(card_x) * sqrt(card_y))
    
    return cos_sim

def rank_doc_sim(rep_file, method, test_path, bc_dict):
    sim_doc_list = []
    doc_vec_list = []
    feature_rep_list = []
    rep_vec = []

    # filename list:
    file_name_list = get_all_files(test_path)
    
    # reading the rep file:
    with open(rep_file) as feature_space_rep:
        for line in feature_space_rep:
            (token, vec_value) = line.split()
            feature_rep_list.append(token)
            rep_vec.append(float(vec_value))

    # creating a feature space:
    feature_space = create_feature_space(feature_rep_list)

    # reading the test_path data as either tokens or cluster_ids
    if method == 'mi':
        token_list = get_doc_wise_tokens(test_path)
    else:
        token_list = load_docwise_collection_clusters(test_path, bc_dict)
    
    # vectorize all the docs in token_list
    for doc in token_list:
        doc_vec_list.append(vectorize(feature_space, doc))
    
    count = 0
    doc_info = []
    for doc_vec in doc_vec_list:
        doc_info = []
        doc_info.append(file_name_list[count])
        count += 1
        doc_info.append(cosine_similarity(rep_vec, doc_vec))
        sim_doc_list.append(tuple(doc_info))

    sim_doc_list.sort(key = itemgetter(1), reverse = True)
    return sim_doc_list

def write_precision_results(mi_doc_list, tfidf_doc_list, rep_list):
    mi_precision = get_precision(mi_doc_list, rep_list)
    tfidf_precision = get_precision(tfidf_doc_list, rep_list)

    outputfile = open('result.txt', 'a')
    outputfile.write(str(mi_precision) +'\n')
    outputfile.write(str(tfidf_precision) +'\n')
    outputfile.close()


def main():
    # for 1a:
    # print(get_all_files(dhruvils_corpus_root))

    # for 1b:
    # print(load_file_tokens('/home1/c/cis530/hw1/data/corpus/illumina/5095882.txt'))

    # for 1c:
    # print(load_collection_tokens(dhruvils_corpus_root))

    # for 2.1a:
    # print(get_tf(load_collection_tokens(dhruvils_corpus_root +'/starbucks')))

    # for 2.1b:
    # an itemlist to act as collection of docs
    #itemlist = get_doc_wise_tokens('/home1/c/cis530/hw1/data/all_data/')
    # precalculate the doc frequencies:
    #calc_doc_freq(itemlist)
    # then calculate the idf for each word:
    # print(get_idf(itemlist))
    
    # for 2.1c:
    #tf_dict = get_tf(load_collection_tokens(dhruvils_corpus_root +'/starbucks'))
    #idf_dict = get_idf(itemlist)
    #print(idf_dict)
    # print(get_tfidf_top(tf_dict, idf_dict, 100))

    # for 2.2a:
    # bg_terms = load_collection_tokens(dhruvils_corpus_root)
    # topic_terms = load_collection_tokens(dhruvils_corpus_root +'/starbucks')
    # print(get_mi_top(bg_terms, topic_terms, 100))
    # print(get_mi_dict(bg_terms, topic_terms))

    # for 2.2b:
    #bg_terms = load_collection_tokens(dhruvils_corpus_root)
    #topic_terms = load_collection_tokens(dhruvils_corpus_root +'/starbucks')
    #write_mi_weights(dhruvils_corpus_root +'/starbucks', 'starbucks_mi_weights.txt')

    # for 2.3:
    #bg_terms = load_collection_tokens(dhruvils_corpus_root)
    #topic_terms = load_collection_tokens(dhruvils_corpus_root +'/starbucks')
    #itemlist = get_doc_wise_tokens('/home1/c/cis530/hw1/data/all_data')
    #calc_doc_freq(itemlist)

    #tf_dict = get_tf(topic_terms)
    #idf_dict = get_idf(itemlist)
    #top_tfidf_list = get_tfidf_top(tf_dict, idf_dict, 50)

    #top_norm_tf_list = get_tf_top(topic_terms, 100)

    #top_mi_list = get_mi_top(bg_terms, topic_terms, 100)
    
    #write_comp_results(top_mi_list, top_norm_tf_list, top_tfidf_list)
    #print(get_precision(top_mi_list, top_tfidf_list))
    #print(get_precision(top_norm_tf_list, top_tfidf_list))

    #print('TFIDF: \t' +str(top_tfidf_list))
    #print('MI LIST: \t' +str(top_mi_list))
    #print('TF LIST: \t' +str(top_norm_tf_list))

    #for 3.1a:
    # print(read_brown_cluster())

    # for 3.1b:
    # bc_dict = read_brown_cluster()
    # print(load_file_clusters(dhruvils_corpus_root +'/starbucks/3213077.txt', bc_dict))

    # for 3.1c:
    # bc_dict = read_brown_cluster()
    # print(load_collection_clusters(dhruvils_corpus_root +'/starbucks', bc_dict))

    # for 3.1d:
    # bc_dict = read_brown_cluster()
    # print(get_idf_clusters(bc_dict))

    # for 3.1e:
    #bc_dict = read_brown_cluster()
    #write_tfidf_weights(dhruvils_corpus_root +'/starbucks', 'starbucks_tfidf_weights.txt', bc_dict)
    # itemlist = load_collection_clusters(dhruvils_corpus_root , bc_dict)
    # print(get_tf_clusters(itemlist))

    # for 4.1a:
    # bc_dict = read_brown_cluster()
    # clusters_list = load_collection_clusters(dhruvils_corpus_root +'/starbucks', bc_dict)
    # print(create_feature_space(clusters_list))

    # for 4.1b:
    # bc_dict = read_brown_cluster()
    # clusters_list = load_collection_clusters('/home1/c/cis530/hw1/data/all_data', bc_dict)
    # feature_space = create_feature_space(clusters_list)
    # cluster_id_list = load_collection_clusters(dhruvils_corpus_root +'/starbucks', bc_dict)
    # vectorize(feature_space, cluster_id_list)

    # for 4.2:
    # bc_dict = read_brown_cluster()                                            
    # clusters_list = load_collection_clusters('/home1/c/cis530/hw1/data/all_data', bc_dict)                                                                  
    # feature_space = create_feature_space(clusters_list)                      
    # doc1_cid_list = load_file_clusters(dhruvils_corpus_root +'/starbucks/2598718.txt', bc_dict)
    # doc2_cid_list = load_file_clusters(dhruvils_corpus_root +'/starbucks/3310796.txt', bc_dict)
    # doc1_vec = vectorize(feature_space, doc1_cid_list)
    # doc2_vec = vectorize(feature_space, doc2_cid_list)
    # print(cosine_similarity(doc1_vec, doc2_vec))

    # for 4.3a:
    bc_dict = read_brown_cluster()
    method_tfidf = 'tfidf'
    method_mi = 'mi'
    test_path = '/home1/c/cis530/hw1/data/mixed'
    rep_file_tfidf = 'starbucks_tfidf_weights.txt'
    rep_file_mi = 'starbucks_mi_weights.txt'
    doc_sim_list_tfidf = rank_doc_sim(rep_file_tfidf, method_tfidf, test_path, bc_dict)
    doc_sim_list_mi = rank_doc_sim(rep_file_mi, method_mi, test_path, bc_dict)
    
    # for 4.3b:
    # all of the above plus this below:
    top_sim_list_tfidf = []
    count = 0
    for doc in doc_sim_list_tfidf:
        if count < 100:
            top_sim_list_tfidf.append(doc[0])
            count += 1

    top_sim_list_mi = []
    count = 0
    for doc in doc_sim_list_mi:
        if count < 100:
            top_sim_list_mi.append(doc[0])
            count += 1

    top_sim_list_tfidf = tuple(top_sim_list_tfidf)
    top_sim_list_mi = tuple(top_sim_list_mi)

    file_list = get_all_files(test_path)
    
    starbucks_file_list = []
    for title in file_list:
        if title.startswith('starbucks'):
            starbucks_file_list.append(title)

    print(get_precision(top_sim_list_tfidf, starbucks_file_list))
    print(get_precision(top_sim_list_mi, starbucks_file_list))
    
    write_precision_results(top_sim_list_mi, top_sim_list_tfidf, starbucks_file_list)

if __name__ == "__main__":
    main()
