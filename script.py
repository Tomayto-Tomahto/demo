from flask import Flask, render_template, request, redirect, url_for, jsonify
import sklearn as skl
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import librosa, librosa.display
from scipy import stats

app=Flask(__name__)
count = 0

s_list = "music a the x back email could center item national"
list = s_list.split(" ")

class SVM:

	prediction = "none"
	wordwrong = ""

	def chunk(self, xs, n):
		'''Split the list, xs, into n evenly sized chunks'''
		L = len(xs)
		s, r = divmod(L, n)
		t = s + 1
		return ([xs[p:p+t] for p in range(0, r*t, t)] +
				[xs[p:p+s] for p in range(r*t, L, s)])

	def detectOnsets(self, y, sr, s=0.12, e=0.08):
		start = 0
		end = len(y) - 1

		for i in range(len(y)):
			counter = 0
			
			if abs(y[i]) > 0.08:
				for j in y[i:i+200]:
					if abs(j) > 0.08:
						counter+=1
						
				if counter > 20:
					start = i 
					break
					

		yy = np.asfortranarray(y[::-1])

		for i in range(len(yy)):
			counter = 0
			if abs(yy[i]) > 0.05:
				for j in yy[i:i+200]:
					if abs(j) > 0.05:
						counter+=1
				if counter > 20:
					end = len(y) - i 
					break
		
		total_time = librosa.get_duration(y=y, sr=sr)
		a = total_time*(start)/len(y)
		b = total_time*(end)/len(y)
		return start,end,total_time, a, b

	def calculateFeatures(self, word):
		y, sr = librosa.load('recording.mp3')
		start,end,total_time, a, b = self.detectOnsets(y,sr)

		# plot
		plt.figure()
		plt.title("Your Pronunciation")
		librosa.display.waveplot(y, sr, alpha=0.8)
		plt.vlines(a, 0, 1, color='r', alpha=0.9,linestyle='--', label='Start')
		plt.vlines(b, 0, 1, color='g', alpha=0.9,linestyle='--', label='End')
		plt.legend()

		# generate unique plot id
		self.pltid = str(uuid.uuid1())
		plt.savefig('static\\plots\\plt'+self.pltid+'.png',transparent=True)
		plt.cla()
		y=y[start:end]

		# calculate features
		chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=3)
		chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=3)
		chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=3)
		rmse = librosa.feature.rms(y=y)
		spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
		spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
		rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
		zcr = librosa.feature.zero_crossing_rate(y)

		y, sr = librosa.load('recording.mp3', sr = 66150)
		start = len(y)*a/total_time
		end = len(y)*b/total_time
		y = y[int(start):int(end)]

		mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 13)

		n = len(word) + 1

		for k in range(13):
			if len(mfcc[k]) < n:
				print(word)
				return

		mfcc1 = self.chunk(mfcc[0], n)
		mfcc2 = self.chunk(mfcc[1], n)
		mfcc3 = self.chunk(mfcc[2], n)
		mfcc4 = self.chunk(mfcc[3], n)
		mfcc5 = self.chunk(mfcc[4], n)
		mfcc6 = self.chunk(mfcc[5], n)
		mfcc7 = self.chunk(mfcc[6], n)
		mfcc8 = self.chunk(mfcc[7], n)
		mfcc9 = self.chunk(mfcc[8], n)
		mfcc10 = self.chunk(mfcc[9], n)
		mfcc11 = self.chunk(mfcc[10], n)
		mfcc12 = self.chunk(mfcc[11], n)
		mfcc13 = self.chunk(mfcc[12], n)

		new_mfcc = mfcc1 + mfcc2 + mfcc3 + mfcc4 + mfcc5 + mfcc6 + mfcc7 + mfcc8 + mfcc9 + mfcc10 + mfcc11 + mfcc12 + mfcc13

		features = []
		flist = [new_mfcc, chroma_stft, chroma_cqt, chroma_cens]
		for feature in flist:
			for e in feature:
				features.append(np.mean(e))
			for e in feature:
				features.append(np.std(e))
			for e in feature:
				features.append(stats.skew(e))
			for e in feature:
				features.append(stats.kurtosis(e))
			for e in feature:
				features.append(np.median(e))
			for e in feature:
				features.append(np.min(e))
			for e in feature:
				features.append(np.max(e))

		feature = [rmse, spec_cent, spec_bw, rolloff, zcr]
		for e in feature:
			features.append(np.mean(e))
			features.append(np.std(e))
			features.append(np.median(e))
			features.append(np.min(e))
			features.append(np.max(e))

			
		header = []
		flist = [new_mfcc, chroma_stft, chroma_cqt, chroma_cens]
		xd = ["mfcc", "chroma_stft", "chroma_cqt", "chroma_cens"]
		count = 0
		for feature in flist:
			count2 = 0
			for e in feature:
				count2 += 1
				header.append(xd[count]+str(count2))
				count2 += 1
				header.append(xd[count]+str(count2))
				count2 += 1
				header.append(xd[count]+str(count2))
				count2 += 1
				header.append(xd[count]+str(count2))
				count2 += 1
				header.append(xd[count]+str(count2))
				count2 += 1
				header.append(xd[count]+str(count2))
				count2 += 1
				header.append(xd[count]+str(count2))
			count += 1

		feature = [rmse, spec_cent, spec_bw, rolloff, zcr]
		count = 0
		xd = ["rmse","spectral_centroid", "spectral_bandwidth", "rolloff", "zero_crossing_rate"]
		for e in feature:
			header.append(xd[count]+"1")
			header.append(xd[count]+"2")
			header.append(xd[count]+"3")
			header.append(xd[count]+"4")
			header.append(xd[count]+"5")
			count += 1
			

		
		
		df = pd.DataFrame(columns=header)
		dict = {}
		count = 0
		for i in header:
			dict[i] = features[count]
			count += 1
			
		df = df.append(dict,ignore_index=True)
		
		# PLOT EXAMPLE
		y, sr = librosa.load('static/data/en/'+word+'/en-US-Wavenet-D0.mp3')
		start,end,total_time, h, hh = self.detectOnsets(y,sr,0.0,0.0)

		# plot
		# plt.figure(figsize=(15, 5))
		plt.figure()
		plt.title("Correct Pronunciation")
		librosa.display.waveplot(y, sr, alpha=0.8)
		plt.vlines(total_time*(start)/len(y), 0, 1, color='r', alpha=0.9,linestyle='--', label='Start')
		plt.vlines(total_time*(end)/len(y), 0, 1, color='g', alpha=0.9,linestyle='--', label='End')
		plt.legend()

		# generate unique plot id
		self.pltid2 = str(uuid.uuid1())
		plt.savefig('static\\plots\\plt'+self.pltid2+'.png',transparent=True)
		plt.cla()

		
		# PLOT EXAMPLE 2
		y, sr = librosa.load('static/data/tr/'+word+'/tr-TR-Wavenet-D0.mp3', mono=True)
		start,end,total_time,h, hh = self.detectOnsets(y,sr,0.0,0.0)

		# plot
		plt.figure()
		plt.title("Wrong Pronunciation")
		librosa.display.waveplot(y, sr, alpha=0.8)
		plt.vlines(total_time*(start)/len(y), 0, 1, color='r', alpha=0.9,linestyle='--', label='Start')
		plt.vlines(total_time*(end)/len(y), 0, 1, color='g', alpha=0.9,linestyle='--', label='End')
		plt.legend()

		# generate unique plot id
		self.pltid3 = str(uuid.uuid1())
		plt.savefig('static\\plots\\plt'+self.pltid3+'.png', transparent=True)
		plt.cla()
		return df

	def datasplit(self, word, features, a1):
		X_train=pd.read_csv('static/data/en/'+word+'/26features_v4.csv', sep=',', usecols=a1)
		X_train = X_train.drop([1,4,7,10])
		y_train = pd.Series(np.ones(34))

		tr_df_train = pd.read_csv('static/data/tr/'+word+'/26features_v4.csv', sep=',', usecols=a1)
		X_train = X_train.append(tr_df_train, ignore_index = True)
		y_train = y_train.append(pd.Series(np.zeros(31)))

		selectedfeatures = features[a1]
		scaler = skl.preprocessing.StandardScaler(copy=False)
		scaler.fit_transform(X_train)
		scaler.transform(selectedfeatures)

		return X_train, y_train, selectedfeatures
		
	def predict(self, word):
		n = len(word) + 1		
		a0 = ['mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc11', 'mfcc12', 'mfcc20', 'mfcc30', 'mfcc56', 'mfcc57', 'mfcc68', 'mfcc70', 'mfcc72', 'chroma_stft11', 'chroma_cqt10']

		listoflist = [[] for i in range(n-1)]
		a1 = []
		for i in a0:
			if i[0] == "m":
				count = 0
				for j in range(int(i[4::])*n+1, (int(i[4::])+1)*n+1):
					a1.append("mfcc"+str(j))
					for k in range(n-1):
						if k!=count and count!=k+1:
							listoflist[k].append("mfcc"+str(j))
					count += 1
			else:
				a1.append(i)
				for ll in range(n-1):
					listoflist[ll].append(i)
		
		self.wordwrong = ""
		features = self.calculateFeatures(word)
		
		X_train, y_train, selectedfeatures = self.datasplit(word, features, a1)
				
		model = skl.svm.SVC(kernel='rbf', C=1.2, probability=True).fit(X_train, y_train)		
		self.prediction = int(model.predict_proba(selectedfeatures)[0][1]*100)
		
		for i in range(n-1):
			X_train, y_train, selectedfeatures = self.datasplit(word, features, listoflist[i])
			model = skl.svm.SVC(kernel='rbf', C=1.2, probability=True).fit(X_train, y_train)		
			if int(model.predict_proba(selectedfeatures)[0][1]*100)>self.prediction*1.1:
				self.wordwrong += str(i)+' '+str(i+1)+' '
			
			
			


svm = SVM()


@app.route('/', methods = ['GET','POST'])
def index():
	return render_template('index.html', word=list[count])

@app.route('/predict', methods = ['GET','POST'])
def predict():
	if request.method == 'POST' and request.files['recording'].filename:
		request.files['recording'].save(os.path.join('./recording.mp3'))
		svm.predict(list[count])
	return jsonify(result=str(svm.prediction), word=list[count], next=svm.pltid, next2=svm.pltid2, next3=svm.pltid3, wordwrong=svm.wordwrong)

@app.route('/skip', methods = ['GET'])
def skip():
	global count
	count+=1
	return redirect(url_for('index'))


if __name__ == '__main__':
	app.run(debug = True)

