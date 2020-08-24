#Import Dataset
library(xlsx)
data=read.xlsx(file.choose(),sheetName = "prabowodebatfinal 6")
View(data)

#Data Cleansing and Preprocessing
library(tm)
library(textclean)
library(tokenizers)
library(dplyr)
tweets <- data$text %>% 
  as.character()
head(tweets,2)

tweets=gsub("http.*"," ",tweets)
tweets=gsub("https.*"," ",tweets)
tweets <- gsub( "\n"," ",tweets)
tweets=tweets%>%
  replace_html() %>%
  replace_url()

tweets=gsub("\\â","..",tweets)
tweets=gsub("\\¦","..",tweets)
tweets=gsub("\\'","..",tweets)
tweets=gsub("RT","",tweets)
tweets=tweets %>%
  replace_emoji(.) %>%
  replace_html(.)

tweets <- tweets %>% 
  replace_tag(tweets, pattern = "@([A-Za-z0-9_]+)",replacement="") %>%  # remove mentions
  replace_hash(tweets, pattern = "#([A-Za-z0-9_]+)",replacement="")      # remove hashtags

tweets <- strip(tweets)

tweets <- as.character(tweets)

#Stemming
library(katadasaR)
stemming <- function(x){
  paste(lapply(x,katadasar),collapse = " ")}
tweets2 <- lapply(tokenize_words(tweets[]), stemming)
tweets <- as.character(tweets2)
corpus=Corpus(VectorSource(tweets))
library(tm)

#Remove stopwords
stopword=read.table(file.choose(),header=FALSE,sep=",")
stopword=as.character(stopword$V1)
no_stopword=tm_map(corpus,removeWords,stopword)

#tokenization
tokenizer=function(x) strsplit(x,split=' ')
token=tm_map(no_stopword,tokenizer)

#Extraction Feature
#TDIDF
weight_tfidf=function(x) weightTfIdf(x)
dtmTfidf=DocumentTermMatrix(token,control=list(weighting=weight_tfidf))

#menghindari overfitting,mengurangi kata-kata yang frekuensinya sedikit
dtm1=removeSparseTerms(dtmTfidf,0.99)
inspect(dtm1)
m=as.matrix(dtm1)

#Visualisasi Token
library(ggplot2)
#kata paling sering muncul
freq=sort(colSums(m),decreasing=T)
head(freq,20)
wf=data.frame(word=names(freq),freq=freq)
ggplot(subset(wf,freq>300),aes(x=reorder(word,-freq),y=freq))+
  geom_bar(stat="identity")

#membuat wordcloud Total
library(wordcloud)
win.graph()
wordcloud(names(freq),freq=freq,min.freq=25,random.order=T,colors=brewer.pal(8,'Dark2'))
wordcloud(names(freq),freq=freq,min.freq=30,random.order=T)


#Sentimen analisis
#Partition Data
reviewPositif=m[which(data$komentar=='Positif'),]
reviewNegatif=m[which(data$komentar=='Negatif'),]
freqPositif=sort(colSums(reviewPositif),decreasing = T)
freqNegatif=sort(colSums(reviewNegatif),decreasing = T)
win.graph()

#Wordcloud Positif dan Negatif
wordcloud(names(freqPositif),freq=freqPositif,min.freq=35,random.order=T,colors=brewer.pal(8,'Dark2'))
win.graph()
wordcloud(names(freqNegatif),freq=freqNegatif,min.freq=10,random.order=T,colors=brewer.pal(8,'Dark2'))
wf2=data.frame(word=names(freqPositif),freq=freqPositif)
wf3=data.frame(word=names(freqNegatif),freq=freqNegatif)
ggplot(subset(wf2,freq>300),aes(x=reorder(word,-freq),y=freq))+
  geom_bar(stat="identity")
win.graph()
ggplot(subset(wf3,freq>300),aes(x=reorder(word,-freq),y=freq))+
  geom_bar(stat="identity")

#membangun model klasifikasi
#data training dan testing
library(caret)
library(e1071)
training_Index=createDataPartition(y=data$Komentar,p=0.7,list=FALSE)
training=cbind(as.data.frame(m[training_Index,]),sentiment=data$Komentar[training_Index])
xtest=as.data.frame(m[-training_Index,])
ytest=as.factor(data$Komentar[-training_Index])

#svm
library(kernlab)
library(e1071)

model_svm=svm(training$sentiment~.,data=training,kernel="rbf",cost=0.01,gamma=0.5)
prediksi=predict(model_svm,xtest)
confusionMatrix(model_svm$fitted,training$sentiment,mode="everything")
confusionMatrix(prediksi,ytest,mode="everything")
#Tuning Hyperparameter
tune.out <- tune(svm, training[,-97],training$sentiment, data =training, kernel = "rbf",
                 ranges = list(cost = c(0.01, 0.1, 1, 5, 10),gamma=c(0.01,0.1,0.5,1,5,10)))
bestmod <- tune.out$best.model
ypred <- predict(bestmod, xtest)
confusionMatrix(ypred,ytest,mode="everything")

#Naive Bayes
model=naiveBayes(formula=training$sentiment~.,data=training)
prediksi=predict(model,xtest)
confusionMatrix(prediksi,ytest,mode="everything")