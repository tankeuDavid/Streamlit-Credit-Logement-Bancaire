import streamlit as st
import pandas as pd
import numpy as no
import pickle


st.write("#l'application qui predit l'accord du credit selon le profil du client")
#    #(diese) pour que le texte soit en gras


#prenons les informations du client kon veut savoir si on l'accordera un credit ou pas

st.sidebar.header("Les caracteristiques du client")

def caracte_entree_client():
# selectbox car ceux sont des variables categoriques
    Gender = st.sidebar.selectbox('Sexe', ('Male', 'Female'))
    Married = st.sidebar.selectbox('Marié', ('Yes', 'No'))
    Dependents = st.sidebar.selectbox('Enfants', ('0', '1', '2', '3+'))
    Education = st.sidebar.selectbox('Education', ('Graduate', 'Not Graduate'))
    Self_Employed = st.sidebar.selectbox('Salarié ou Entrepreneur', ('Yes', 'No'))
#slider car ceux sont des variables numeriques
    ApplicantIncome = st.sidebar.slider('Salaire du client', 150, 4000, 200)
# 15O car le salaire min est 150 et le salaire max est 4000 le voir a partir de jupithenotebook en utilisant la methode min()
    CoapplicantIncome = st.sidebar.slider('Salaire du conjoint', 0, 40000, 2000)
#le salaire min du conjoint est 0 celui max est 40000 et le salaire par defaut est 2000
    LoanAmount = st.sidebar.slider('Montant du crédit en Kdollar', 9.0, 700.0, 200.0)
    Loan_Amount_Term = st.sidebar.selectbox('Durée du crédit',
                                            (360.0, 120.0, 240.0, 180.0, 60.0, 300.0, 36.0, 84.0, 12.0))
    Credit_History = st.sidebar.selectbox('Credit_History', (1.0, 0.0))
    Property_Area = st.sidebar.selectbox('Property_Area', ('Urban', 'Rural', 'Semiurban'))


# Definissons maintenant les cles de notre dictionnaires

#notre dictonnaire a ete cree car les valeurs kon va recuperer entree par utilsateur on va les injecter dans les valeurs de notre dictionnaire

# dans notre dictionnaire on'a les cles et le contenu des valeurs(importe du haut entree par le client
# dans ce dictionnaire la 1ere valeurs c'est les donnes du client


    data = {
    'Gender':Gender,
    'Married' :Married,
    'Dependents' :Dependents,
    'Education' :Education,
    'Self_Employed' :Self_Employed,
    'ApplicantIncome' :ApplicantIncome,
    'CoapplicantIncome' :CoapplicantIncome,
    'LoanAmount' :LoanAmount,
    'Loan_Amount_Term' :Loan_Amount_Term,
    'Credit_History' :Credit_History,
    'Property_Area' :Property_Area

}

# ce kon 'a dans ce dictionnaire on va le mettre sous forme de DataFrame ou encore de base de donnee

# La fonction DataFrame permet de rendre un dictionnaire sous forme de base de donne

    profil_client = pd.DataFrame(data,index=[0])
    return  profil_client

input_dataFrame = caracte_entree_client()



# transformer les donnes d'entree en donnees adaptes au modele(comme on'a suprime la colone load_status, on na utilise la methode getdummies()pour transformer toutes les chaines de caractere en chiffre

# Chargeons d'abord notre BD pour le faire

df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
# supprimons les deux colonnes kon utilise pas
credit_input=df.drop(columns=['Loan_ID','Loan_Status'])

donnee_entree = pd.concat([input_dataFrame,credit_input], axis=0)

#faisons maintenant un encodege ou une preparation des donnes

# encodage des données
var_cat=['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History', 'Property_Area']
for col in var_cat:
    dummy=pd.get_dummies(donnee_entree[col],drop_first=True)
    donnee_entree=pd.concat([dummy,donnee_entree],axis=1)
    del donnee_entree[col]
#prendre uniquement la premiere ligne
donnee_entree=donnee_entree[:1]

#afficher les données transformées
st.subheader('Les caracteristiques transformés')
st.write(donnee_entree)


#importer le modele

load_model = pickle.loads(open('Prevision_credit_logement_bancaire.pkl' , 'rb'))

#appliquer le modele sur le profil d'entree

prevision = load_model.predict(donnee_entree)

st.subheader('Resultat de la prevision')
st.write(prevision)