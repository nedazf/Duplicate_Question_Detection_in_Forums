import csv
datasets = ["android", "apple", "askubuntu", "quora", "sprint" , "superuser"]
for  dataset in datasets:
	path = "data/"+dataset +"/combined/"
	full= path+"full_converted.csv"
	train= path+"train_converted.csv"
	test= path +"test_converted.csv"

	with open(full) as myfile:
		reader = csv.reader(myfile, delimiter=',')
		count = 0
		for row in reader:
			count +=1

	nesbat = 4./5

	trainsize = int (nesbat* count)


	trainfile= open(train, "wb" )
	trainwriter = csv.writer(trainfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
	trainwriter.writerow(["id", "qid1", "qid2", "question1", "question2", "is_duplicate"])

	testfile= open(test, "wb" )
	testwriter = csv.writer(testfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
	testwriter.writerow(["id", "qid1", "qid2", "question1", "question2", "is_duplicate"])

	with open(full) as myfile:
		reader = csv.reader(myfile, delimiter=',')
		count = 0
		for row in reader:
			if count  ==0 :
				count+=1
				continue
			if count < trainsize :
				trainwriter.writerow([row[0] , row[1] , row[2] , row[3], row[4], row[5]] )
			else:
				testwriter.writerow(row)
			count +=1
