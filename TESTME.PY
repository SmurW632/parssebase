import spacy

nlp = spacy.load("Model")
#СЮДА ПОСТУПАЕТ ПРЕДЛОЖЕНИЕ ПРЕДПОЛАГАЕМОЕ
promt = ""
doc = nlp(promt)

userneedit = [] #ТУТ СОЗДАЁТСЯ МАССИВ ТРЕБОВАНИЙ ПОЛЬЗОВАТЕЛЯ
for token in doc.ents:
    if "ПОМЕЩЕНИЕ" not in userneedit:
        if str(token.label_) == "ПОМЕЩЕНИЕ":
            userneedit.append("ПОМЕЩЕНИЕ")
    if "МЕРЫ ПОДДЕРЖКИ" not in userneedit:
        if str(token.label_) == "МЕРЫ ПОДДЕРЖКИ":
            userneedit.append("МЕРЫ ПОДДЕРЖКИ")
    if "33" not in userneedit:
        if str(token.label_) == "33":
            userneedit.append("33")
    if "32" not in userneedit:
        if str(token.label_) == "32":
            userneedit.append("32")
    if "31" not in userneedit:
        if str(token.label_) == "31":
            userneedit.append("31")
    if "30" not in userneedit:
        if str(token.label_) == "30":
            userneedit.append("30")
    if "29" not in userneedit:
        if str(token.label_) == "29":
            userneedit.append("29")






    if "21" not in userneedit:
        if str(token.label_) == "21":
            userneedit.append("21")
    if "20" not in userneedit:
        if str(token.label_) == "20":
            userneedit.append("20")
    if "19" not in userneedit: 
        if str(token.label_) == "19":
            userneedit.append("19")
print(userneedit, [(x.label_,x.text) for x in doc.ents])



