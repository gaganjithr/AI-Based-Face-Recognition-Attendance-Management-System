from	sklearn.datasets	import	load_iris	
from	sklearn.neighbors	import	KNeighborsClassifier	
from	sklearn.model_selection	import	train_test_split	
from	sklearn.metrics	import	accuracy_score	
X,	y	=	load_iris(return_X_y=True)	
X_train,	X_test,	y_train,	y_test	=	train_test_split(X,	y,	
test_size=0.4,	random_state=21)	
model	=	KNeighborsClassifier(n_neighbors=3)	
model.fit(X_train,	y_train)	
pred	=	model.predict(X_test)	
for	t,	p	in	zip(y_test,	pred):	
    print(f"Target	=	{t},	Predicted	=	{p}")	
print("Accuracy:",	accuracy_score(y_test,	pred))