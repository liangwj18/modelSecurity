import copy
import random


class Sentence():
    def __init__(self, example:list):
        self.combinations = []
        for part in example:
            self.combinations.append([part])

        self.SS_attribute_index = -1
        self.SS_anti_words = []
        self.SS_irrelevant_words = []

        self.intervention_subj1_index = -1
        self.intervention_subj2_index = -1

        self.response_mask_index = -1
        self.response_p_act_index = -1

    def add(self, index, words:list):
        for word in words:
            self.combinations[index].append(word)

    def negative_expression(self, subject_index, object_index, descrption_index, descrption_list):
        self.combinations[subject_index], self.combinations[object_index] = self.combinations[object_index], self.combinations[subject_index]
        self.combinations[descrption_index] = descrption_list

    def generate_SS_format(self):
        assert self.SS_attribute_index != -1
        output = []
        # print(self.combinations)
        index_bi_character = self.combinations.index(['比'])
        index_B = index_bi_character + 1

        temp_attribute_index = -1
        temp_combinations = []
        for i in range(len(self.combinations)):
            if i == index_bi_character or i == index_B:
                continue
            temp_combinations.append([])
            if i == self.SS_attribute_index:
                temp_attribute_index = len(temp_combinations) - 1
                continue
            for k in range(len(self.combinations[i])):
                part = self.combinations[i][k]
                part = part.replace('更','比较')
                temp_combinations[-1].append(part)

        # print(temp_combinations)
        for i in range(len(self.combinations[self.SS_attribute_index])):

            stereo_word = self.combinations[self.SS_attribute_index][i]
            if len(stereo_word) > 4:
                continue
            anti_word = random.choice(self.SS_anti_words[len(stereo_word) - 2])
            irrelevant_word = random.choice(self.SS_irrelevant_words[len(stereo_word) - 2])

            temp_sentences = []
            for word in [stereo_word, anti_word, irrelevant_word]:
                
                temp_combinations[temp_attribute_index] = [word]
                # print(temp_combinations)
                temp_sentences.append(self._generate(temp_combinations))
            
            for k in range(len(temp_sentences[0])):

                index = temp_sentences[0][k].find(stereo_word)
                output.append([[temp_sentences[0][k], temp_sentences[1][k], temp_sentences[2][k]], [index, index+len(stereo_word)]])

        return output
    def generate_CrowsP_format(self):
        
        output = []
        # print(self.combinations)
        index_bi_character = self.combinations.index(['比'])
        index_A = index_bi_character - 1
        index_B = index_bi_character + 1

        temp_index_A = -1
        temp_combinations = []
        for i in range(len(self.combinations)):
            if i == index_bi_character or i == index_B:
                continue
            temp_combinations.append([])
            if i == index_A:
                temp_index_A = len(temp_combinations) - 1
                continue
            for k in range(len(self.combinations[i])):
                part = self.combinations[i][k]
                part = part.replace('更','比较')
                temp_combinations[-1].append(part)

        # print(temp_combinations)
        for i in range(min(len(self.combinations[index_A]), len(self.combinations[index_B]))):

            A_word = self.combinations[index_A][i]
            B_word = self.combinations[index_B][i]

            if len(A_word) != len(B_word): 
                continue

            temp_combinations[temp_index_A] = [A_word]
            # print(temp_combinations)
            A_sentences = self._generate(temp_combinations)
            # print(A_sentences)

            temp_combinations[temp_index_A] = [B_word]
            B_sentences = self._generate(temp_combinations)

            for k in range(len(A_sentences)):
                index = A_sentences[k].find(A_word)
                output.append([[A_sentences[k], B_sentences[k]], [index, index + len(A_word)]])

        return output

    def generate_Intervention_format(self):

        output = []
        iters = []
        size = 1
        combinations = self.combinations
        for i, combination in enumerate(combinations):
            iters.append(0)
            size *= len(combination)
        for i in range(size):
            sen = ""
            sen_pair = ""
            word1 = ""
            word2 = ""
            for k in range(len(combinations)):
                sen += combinations[k][iters[k]]
                if k == self.intervention_subj1_index:
                    word2 = combinations[self.intervention_subj2_index][iters[self.intervention_subj2_index]]
                    sen_pair += word2
                elif k == self.intervention_subj2_index:
                    word1 = combinations[self.intervention_subj1_index][iters[self.intervention_subj1_index]]
                    sen_pair += word1
                else:
                    sen_pair += combinations[k][iters[k]]

            output.append([[sen, sen_pair], [word1, word2]])

            if i+1 < size:
                iters[-1] += 1
                for k in range(len(combinations)-1, -1 ,-1):
                    if iters[k] == len(combinations[k]):
                        iters[k] = 0
                        iters[k-1] += 1
        return output


    def _generate(self, combinations):
        output = []
        iters = []
        size = 1
        for i, combination in enumerate(combinations):
            iters.append(0)
            size *= len(combination)
        for i in range(size):
            sen = ""
            for k in range(len(combinations)):
                sen += combinations[k][iters[k]]
            output.append(sen)

            if i+1 < size:
                iters[-1] += 1
                for k in range(len(combinations)-1, -1 ,-1):
                    if iters[k] == len(combinations[k]):
                        iters[k] = 0
                        iters[k-1] += 1
        return output

    def generate(self):
        return self._generate(self.combinations)

class QA():
    def __init__(self, context:Sentence,
                        response_pair:list[Sentence],
                        interventions:list[str]):
        self.context = context
        self.response_pair = response_pair
        self.interventions = interventions

    def generate(self):

        context_list = self.context.generate_Intervention_format()
        for context in context_list:
            context_str = []
            for context_index in range(2):
                context_pair = []
                for response in response_pair:
                    
                    for intervention in interventions:
                        st = context[0][context_index] + " " + intervention + " " + res
