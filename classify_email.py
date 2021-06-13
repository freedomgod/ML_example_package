import pandas as pd
import numpy as np
import jieba


class Mail_classify:
    @staticmethod
    def clean_text(st, stopws=None):  # 清理句子中的停用词
        if stopws is None:
            return st
        c_st = []
        for x in st:
            x = x.strip()
            if x not in stopws:
                c_st.append(x)
        return c_st

    def probability_sample(self, sample, data, stopws=None):   # 计算含有sample词语的先验概率
        count = 0
        for x in data['content']:
            data_jb = jieba.lcut(x)
            if stopws is not None:
                data_res = self.clean_text(data_jb, stopws=stopws)
                if sample in data_res:
                    count += 1
            else:
                if sample in data_jb:
                    count += 1
        return count / len(data)

    def fit(self, sample, data, stopws=None):  # 拟合sample句子为垃圾邮件的概率
        sam_jb = jieba.lcut(sample)  # 对邮件内容进行分词
        sam_res = self.clean_text(sam_jb, stopws=stopws)  # 进行文本清理，去除空格、标点以及停用词
        N = len(sam_res)
        P = []
        p1 = len(data[data['is_spam'] == 1]) / len(data)  # 计算训练数据集中垃圾邮件的概率
        for x in sam_res:
            p_sample = self.probability_sample(x, data, stopws=stopws)  # 计算词语x在数据集中的概率
            data1 = data[data['is_spam'] == 1]
            p_sample_1 = self.probability_sample(x, data1, stopws=stopws)  # 计算为垃圾邮件时x出现的后验概率
            try:
                P_1 = (p_sample_1 * p1) / p_sample  # 计算为垃圾邮件的概率
            except ZeroDivisionError:
                N -= 1
                continue
            P.append(P_1)
        return np.sum(P) / N  # 最后的概率为所有概率的均值

    def fit_instance(self, spam_path, ham_path, stopwords_path):
        email_info = pd.DataFrame(columns=['content', 'is_spam'])
        for x in range(1, 26):
            with open(f'{spam_path}/{x}.txt', 'r') as fp:
                email_info = email_info.append({'content': fp.read(), 'is_spam': 1}, ignore_index=True)
            with open(f'{ham_path}/{x}.txt', 'r', errors='ignore') as ff:
                email_info = email_info.append({'content': ff.read(), 'is_spam': 0}, ignore_index=True)

        with open(f'{stopwords_path}', 'r') as fp:
            stopws = fp.read()
        stopws = stopws.split('\n')
        stopws = [s.strip() for s in stopws]

        train_data = email_info.sample(frac=0.8, axis=0)
        test_data = email_info[~email_info.index.isin(train_data.index)]

        tt = pd.DataFrame(columns=['label', 'fit_prob', 'fit_label'])
        for (i, lab) in zip(test_data['content'], test_data['is_spam']):
            fit_p = self.fit(i, train_data, stopws)
            fit_lab = 1 if fit_p > 0.5 else 0
            print(f'邮件标签：{lab}，拟合概率：{fit_p}，拟合标签：{fit_lab}')
            tt = tt.append({'label': lab, 'fit_prob': fit_p, 'fit_label': fit_lab}, ignore_index=True)

        accuracy = len(tt[tt['label'] == tt['fit_label']]) / len(tt)
        return accuracy
