from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import librosa
import sklearn as skl
import os
import numpy as np
import random

app=Flask(__name__)


class SVM:

	prediction = "none"
	def calculateFeatures(self):
		y, sr = librosa.load('recording.mp3', mono=True)
		chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
		rmse = librosa.feature.rms(y=y)
		spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
		spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
		rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
		zcr = librosa.feature.zero_crossing_rate(y)
		mfcc = librosa.feature.mfcc(y=y, sr=sr)
		features = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]    
		for e in mfcc:
			features.append(np.mean(e))
		features=[features]
		scaler = skl.preprocessing.StandardScaler(copy=False)
		scaler.fit_transform(features)
		return features
	
	def predict(self, word):
		model = pickle.load(open('ml/'+word+'.pkl', 'rb'))
		features = self.calculateFeatures()
		self.prediction = (model.predict_proba(features)[0][1]*100)//1
		# %43 native user :(
		#print(model.predict(features))
		#print(model.predict_log_proba(features))
		#print(model.predict_proba(features))

svm = SVM()

list = ['a', 'about', 'add', 'address', 'after', 'all', 'also', 'am', 'an', 
		'and', 'any', 'are', 'as', 'at', 'available', 'b', 'back', 'be', 'because', 
		'been', 'before', 'best', 'book', 'books', 'business', 'but', 'buy', 'by', 
		'c', 'can', 'car', 'center', 'city', 'click', 'comments', 'community', 'company', 
		'contact', 'copyright', 'could', 'd', 'data', 'date', 'day', 'days', 'de', 'design', 
		'details', 'development', 'did', 'do', 'does', 'e', 'each', 'ebay', 'education', 
		'email', 'f', 'find', 'first', 'for', 'free', 'from', 'full', 'games', 'general', 
		'get', 'go', 'good', 'great', 'group', 'had', 'has', 'have', 'he', 'health', 'help', 
		'her', 'here', 'high', 'his', 'home', 'hotel', 'hotels', 'how', 'i', 'if', 'in', 
		'info', 'information', 'international', 'internet', 'into', 'is', 'it', 'item', 
		'items', 'its', 'jan', 'january', 'just', 'know', 'last', 'life', 'like', 'line',
		'links', 'list', 'local', 'm', 'made', 'mail', 'make', 'management', 'many', 'map',
		'may', 'me', 'member', 'message', 'more', 'most', 'music', 'must', 'my', 'n', 'name', 
		'national', 'need', 'new', 'news', 'next', 'no', 'not', 'now', 'number', 'of', 'off', 
		'office', 'on', 'one', 'online', 'only', 'or', 'order', 'other', 'our', 'out', 'over',
		'p', 'page', 'part', 'people', 'please', 'pm', 'policy', 'post', 'posted', 'price', 
		'privacy', 'product', 'products', 'program', 'public', 'r', 're', 'read', 'real', 
		'report', 'research', 'results', 'review', 'reviews', 'right', 'rights', 's', 'said',
		'school', 'search', 'see', 'send', 'service', 'services', 'set', 'sex', 'she', 'should', 
		'site', 'so', 'software', 'some', 'state', 'store', 'such', 'support', 'system', 't',
		'take', 'terms', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 
		'they', 'this', 'those', 'through', 'time', 'to', 'top', 'travel', 'two', 'type', 
		'under', 'united', 'university', 'up', 'us', 'use', 'used', 'user', 'using', 'very',
		'video', 'view', 'was', 'way', 'we', 'web', 'well', 'were', 'what', 'when', 'where',
		'which', 'who', 'will', 'with', 'work', 'world', 'would', 'x', 'year', 'years', 'you', 'your']

random.shuffle(list)
count = 0


@app.route('/', methods = ['GET','POST']) 
def index():
	return render_template('index.html', word=list[count])

@app.route('/predict', methods = ['GET','POST'])
def predict():
	#if request.method == 'POST' and request.files['recording'].filename:
		#request.files['recording'].save(os.path.join('./', 'recording.mp3'))
	svm.predict(list[count])
	return jsonify(result="%"+str(svm.prediction)+" native speaker")
	

@app.route('/skip', methods = ['GET']) 
def skip():
	global count
	count+=1
	return redirect(url_for('index'))


if __name__ == '__main__':
	app.run(debug = True)

