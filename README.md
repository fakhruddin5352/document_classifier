# Document Classifier

Classifies UAE immigration documents into 10 categories. 
1. Birth Certificate
1. EIDA Card Application
1. Employement Contract
1. Entry Permit Visa
1. House Rental Contract
1. Passport
1. UAE Residency
1. Salary Certificate
1. Person Photo

The training data are images in jpeg format.
Vgg19 model has been used with transfer learning.

The prediction api is being served by a wsgi server as a REST endpoint.
