# -*- coding: utf-8 -*-


from elasticsearch import Elasticsearch


# Search only in the ngram field
def elastic_query(query, max_doc):
    es = Elasticsearch()
    if len(query.split(' ')) > 1:
        res = es.search(index='dbpedia', body={'size': max_doc, 'query': {'match_phrase': {'surface_ngram':{'query':query}}}})
    else:
        res = es.search(index='dbpedia', body={'size': max_doc, 'query': {'match': {'surface_ngram':{'query':query}}}})
    return res

# Multi-match in standard and ngram fields 
def elastic_query_multi(query, max_doc, boost):
    es = Elasticsearch()
    max_doc = 100
    # If the query is a phrase, uses match_phrase, otherwise, uses match
    if len(query.split(' ')) > 1:
        res = es.search(index='dbpedia', body={'size': max_doc, 'query': {'multi_match': {'query':query, 'fields': ['surface_form^{}'.format(boost), 'surface_ngram'], 'type':'phrase'}}}) 
    else:
        res = es.search(index='dbpedia', body={'size': max_doc, 'query': {'multi_match': {'query':query, 'fields': ['surface_form^{}'.format(boost), 'surface_ngram']}}})
    return res

# Search entity candidates using the engine elasticsearch
# type_query values = 'single' or 'multi'
def select_candidates(tweet, vocab2idx, type_query = 'multi', max_doc = 100, boost = 3, verbose = 'no'):
    if verbose == 'yes':
        print('\n..:: Entity Candidate Selection ..::')

    # List of list of candidates for each mention
    # Example: [ [dbr:Star_Wars, dbr:Star_Wars:_VII], [dbr:Shane_Helms] ]
    list_candidates = list()
    
    if type_query == 'single':
        for mention in tweet.mentions:
            res = elastic_query(mention[0], max_doc)
            # List of candidates for the current mention
            list_candidate = []
            # If the query on the ElasticSearch has returned nothing, try to split the mention in more words
            if res['hits']['total']['value'] == 0:
                name_words = mention[0].split(' ')
                if len(name_words) > 1:
                    possible_names = list()
                    # Flag to verified if the list_candidates need to receive None at the end
                    flag = False
                    # Try the combination of different names with size -1
                    # This step is done only once
                    # Example: The Force Awakens
                    # Example: ['Force Awakens', 'The Awakens', 'The Force']
                    for i in range(len(name_words)):
                        temp_list = [x for j, x in enumerate(name_words) if j!=i]
                        possible_names.append(' '.join(temp_list))
                    # To be sure that temp_list will be empty for the next candidate
                    temp_list.clear()
                    for name in possible_names:
                        # For every new name, search in the Elasticsearch
                        temp_res = elastic_query(name, max_doc)
                        if temp_res['hist']['total']['value'] > 0:
                            for hit in temp_res['hits']['hits']:
                                candidate = ('dbr:' + hit['_source']['uri'], hit['_source']['uri_count'])
                                #list_candidate.append('dbr:' + hit['_source']['uri'])
                                list_candidate.append(candidate)
                        else:
                            flag = True
                    if flag:
                        list_candidates.append(None)
                    else:
                        list_candidates.append(list_candidate)
                # If nothing was returned by the query and the mention is not a phrase, there is no candidate for it
                else:
                    list_candidates.append(None)
            # If the query returned candidates, put them on the list
            else:
                for hit in res['hits']['hits']:
                    candidate = ('dbr:' + hit['_source']['uri'], hit['_source']['uri_count'])
                    #list_candidate.append('dbr:' + hit['_source']['uri'])
                    list_candidate.append(candidate)
                list_candidates.append(list_candidate)
    # Multi-match query
    else:
        for mention in tweet.mentions:
            res = elastic_query_multi(mention[0], max_doc, boost)
            # List of candidates for the current mention
            list_candidate = []
            # If the query on the ElasticSearch has returned nothing, try to split the mention in more words
            if res['hits']['total']['value'] == 0:
                name_words = mention[0].split(' ')
                if len(name_words) > 1:
                    possible_names = list()
                    # Flag to verified if the list_candidates need to receive None at the end
                    flag = False
                    # Try the combination of different names with size -1
                    # This step is done only once
                    # Example: The Force Awakens
                    # Example: ['Force Awakens', 'The Awakens', 'The Force']
                    for i in range(len(name_words)):
                        temp_list = [x for j, x in enumerate(name_words) if j!=i]
                        possible_names.append(' '.join(temp_list))
                    # To be sure that temp_list will be empty for the next candidate
                    temp_list.clear()
                    for name in possible_names:
                        # For every new name, search in the Elasticsearch
                        temp_res = elastic_query_multi(name, max_doc, boost)
                        if temp_res['hits']['total']['value'] > 0:
                            for hit in temp_res['hits']['hits']:
                                candidate = ('dbr:' + hit['_source']['uri'], hit['_source']['uri_count'])
                                #list_candidate.append('dbr:' + hit['_source']['uri'])
                                list_candidate.append(candidate)
                        else:
                            flag = True
                    if flag:
                        list_candidates.append(None)
                    else:
                        list_candidates.append(list_candidate)
                # If nothing was returned by the query and the mention is not a phrase, there is no candidate for it
                else:
                    list_candidates.append(None)
            # If the query returned candidates, put them on the list
            else:
                for hit in res['hits']['hits']:
                    candidate = ('dbr:' + hit['_source']['uri'], hit['_source']['uri_count'])
                    #list_candidate.append('dbr:' + hit['_source']['uri'])
                    list_candidate.append(candidate)
                list_candidates.append(list_candidate)

    # Making sure that the list returned is correct
    final_candidates = []
    for i, t_candidate in enumerate(list_candidates):
        # If the candidate list for mention i is not empty, append the list
        if t_candidate:
            temp_candidate = []
            for t in t_candidate:
                if t[0].lower() in vocab2idx:
                    temp_candidate.append(t)
            if not temp_candidate:
                final_candidates.append(None)
            else:
                temp_candidate = sorted(temp_candidate, key=lambda x: (-x[1], x[0]))
                if len(temp_candidate) > max_doc:
                    temp_candidate = temp_candidate[:max_doc]
                final_candidates.append(temp_candidate)
        # If the candidate list for mention i is empty, append none
        else:
            final_candidates.append(None)
    if verbose == 'yes':
        for i, candidate in enumerate(final_candidates):
            print('Mention: {}'.format(tweet.mentions[i]))
            if candidate is None:
                print('\tNone')
            else:
                print('\t{}'.format(candidate[0]))
    return final_candidates