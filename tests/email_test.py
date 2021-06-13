from classify_email import Mail_classify

Mc = Mail_classify()
acc = Mc.fit_instance(spam_path='../my_data/spam', ham_path='../my_data/ham',
                      stopwords_path='../my_data/en_stopwords.txt')
print(acc)
