import json
import pandas as pd

with open('yelp_academic_dataset_business.json') as f:
    data = []
    idnamereview = []
    review = []
    
    counter = 0
    terminate = 42153
     
    for line in f:
        data = json.loads(line)
        name = data['name'].encode('utf8')
        buzid = data['business_id'].encode('utf8')
        review = data['review_count']
        idnamereview.append((buzid, name, review))
        
        counter += 1
        if counter == terminate: break

with open ('id_name_review.csv','w') as f:
    for buzid, name, review in idnamereview: f.write(buzid + ',' + '\"' + name + '\"' + ',' + str(review) + '\n')
    



top100 = pd.read_csv('top100.csv')

with open('dataset_review.json','w') as fileOut:
    fileOut.write('[')

    with open ('yelp_academic_dataset_review.json') as file:
        for line in file:
            fileOut.write(line + ',')
    i = fileOut.tell()
    fileOut.seek(i-1)
    fileOut.truncate()
    fileOut.write(']')


result = pd.read_json('dataset_review.json')
result = result.drop([result.columns[1], result.columns[2], result.columns[3], result.columns[5], result.columns[6], result.columns[7]], axis=1)


temp = {idx:[] for idx in top100['id']}
for i in range(len(result['business_id'])):
    if result['business_id'][i] in temp.keys():
        temp[result['business_id'][i]].append(result['text'][i])


ids = temp.keys()
reviews_temp = map(' '.join, temp.values())
reviews = [reviews_temp[ids.index(idx)] for idx in top100['id']]
for i in range(len(reviews)):
    reviews[i] = reviews[i].encode('utf8')
top100['reviews'] = reviews
del top100['id']



for i in range(len(top100['name'])):
    with open('name_review_' + str(i) + '.txt', 'w') as f:
        f.write(top100['name'][i] + ',' + top100['reviews'][i])




