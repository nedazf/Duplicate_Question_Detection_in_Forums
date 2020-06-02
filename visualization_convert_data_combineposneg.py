import csv
import random
class Dataset(object):
	def __init__(self , name, corpusfile, txtfiles):
		self.name= name
		self.corpusfile = corpusfile
		self.txtfiles= txtfiles
		self.corpusDictionary= dict()
	def read_corpusfile(self,corpusfile):
		print " reading corpusfile "  , corpusfile

		with open(corpusfile) as myfile:
			reader = csv.reader(myfile, delimiter='\t')
			for row in reader:

				self.corpusDictionary[row[0] ]=row[1]

	def readandconvert_txtfile(self, txtfile):
		convertedfile = txtfile+ "_converted.csv"
		print "creating convertedfile" , convertedfile
		convertedfile= open(convertedfile, "wb" )
		cwriter = csv.writer(convertedfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

		id = 0
		duplicates = dict()
		mylist= []
		for label  in ["pos" , "neg"]:
			with open(txtfile+"."+label+".txt", "r") as myFile:

				for line in myFile:

					line = line.rstrip('\n')

					line = line.split(" ")
					qid1 = line[0]
					qid2 = line[1]
					question1 = self.corpusDictionary[qid1]
					question2 = self.corpusDictionary[qid2]
					if "neg" == label:
						is_duplicate= 0
					elif "pos"  == label:

						is_duplicate= 1
					if (qid1, qid2) not in duplicates and (qid2, qid1) not in duplicates:
						mylist.append([ str(id), str(qid1), str(qid2), str(question1) , str(question2), str(is_duplicate)])
						id +=1
		random.shuffle(mylist)
		cwriter.writerow(["id", "qid1", "qid2", "question1", "question2", "is_duplicate"])
		for row in mylist :
			cwriter.writerow(row)
		convertedfile.close()

if __name__ == "__main__":
	path = "data/"
	datasets = ["android", "apple", "askubuntu", "quora", "sprint" , "superuser"]

	for dataset in datasets :
		#  first use gunzip on corpus files (gunzip corpus.tsv.gz   results in corpus.tsv ) ,  then run code
		corpusfile= path + dataset +"/" +"corpus.tsv"
		txtfilesnames= [ "dev", "full",  "test" , "train" ]
		txtfiles= []
		for txtfilename in txtfilesnames:
			txtfiles.append(path+dataset+"/"  +txtfilename)


		dataset= Dataset(dataset, corpusfile  , txtfiles )
		try:
			dataset.read_corpusfile(corpusfile)
		except:
			print  "you should probably gunzip the corpusfile, type "  , " gunzip " + corpusfile
		for txtfile in dataset.txtfiles :
			try:
				dataset.readandconvert_txtfile(txtfile)
			except:
				print "skipping, this dataset does not contain ", txtfile