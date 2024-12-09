from gliner import GLiNER


def predict(model, text):
    

    labels = ["person", "book", "location", "date", "actor", "character"]
    entities =model.predict_entities(text, labels, threshold=0.4)

    
    # for entity in entities:
    #     entities1.append(entity["text"], "=>" , entity["labels"])
    return entities
    