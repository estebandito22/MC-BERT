import json

import csv
import os
from collections import Counter

#define all our files, enough of them we're just hardcoding here

train_answers_json = '/beegfs/ijh216/vqa2/Annotations/v2_mscoco_train2014_annotations.json'
val_answers_json = '/beegfs/ijh216/vqa2/Annotations/v2_mscoco_val2014_annotations.json'

train_questions_json = '/beegfs/ijh216/vqa2/Questions/v2_OpenEnded_mscoco_train2014_questions.json'
val_questions_json = '/beegfs/ijh216/vqa2/Questions/v2_OpenEnded_mscoco_val2014_questions.json'
test_dev_questions_json = '/beegfs/ijh216/vqa2/Questions/v2_OpenEnded_mscoco_test-dev2015_questions.json'
test_questions_json = '/beegfs/ijh216/vqa2/Questions/v2_OpenEnded_mscoco_test2015_questions.json'

base_image_dir = '/beegfs/ijh216/vqa2/Images/real/mscoco'


train_meta = 'mscoco_train2014.csv'
val_meta = 'mscoco_val2014.csv'
test_dev_meta = 'mscoco_test_dev2015.csv'
test_meta = 'mscoco_test.csv'

#build our answer lookup
all_answers = []

for jfile in [train_answers_json, val_answers_json]:

    with open(jfile) as f:
        answers = json.load(f)['annotations']

    for answer in answers:
        all_answers.append(answer['multiple_choice_answer'])

token_counter = Counter(all_answers)
vocab, count = zip(*token_counter.most_common(100_000))
id2token = list(vocab)
token2id = dict(zip(vocab, range(len(vocab))))

with open('labels.txt', 'w') as f:
    for tok in id2token:
        f.write(tok + '\n')

#build our actual lookup tables
def build_lookup(jfile):
    answer_lookup = {}

    with open(jfile) as f:
        answers = json.load(f)['annotations']

    for answer in answers:
        ans_value = answer['multiple_choice_answer']
        question_id = answer['question_id']

        answer_lookup[question_id] = token2id[ans_value]

    return answer_lookup

#I think the question IDs are unique, so this could probably have been one but just in case
train_answers_lookup = build_lookup(train_answers_json)
val_answers_lookup = build_lookup(val_answers_json)


#create an image string form a prefix and image_id
def build_image(prefix, image_id):
    return os.path.join(base_image_dir, prefix, "COCO_" + prefix + "_" + str(image_id).zfill(12) + ".jpg")


def write_metadata(input_file, output_file, prefix, lookup):

    with open(input_file) as f:
        questions = json.load(f)['questions']

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)


        for question in questions:

            row = [build_image(prefix, question['image_id']), question['question'],
                   lookup[question['question_id']] if lookup else '']
            writer.writerow(row)


write_metadata(train_questions_json, train_meta, 'train2014', train_answers_lookup)
write_metadata(val_questions_json, val_meta, 'val2014', val_answers_lookup)
write_metadata(test_dev_questions_json, test_dev_meta, 'test2015', None)
write_metadata(test_questions_json, test_meta, 'test2015', None)


