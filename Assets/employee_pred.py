
import pickle

# de-serialize model.pkl file into an object called mdl using pickle
# with open('Assets\pickle\model.pkl', 'rb') as handle:
with open('/opt/render/project/src/Assets/pickle/model.pkl', 'rb') as handle:
    mdl = pickle.load(handle)

# de-serialize scalar.pkl file into an object called scl using pickle
# with open('Assets\pickle\scalar.pkl', 'rb') as handle:
with open('/opt/render/project/src/Assets/pickle/scalar.pkl', 'rb') as handle:
    scl = pickle.load(handle)

# scale the user fed data
def scale(sample):
    return scl.transform(sample)

# make predicion
def predict(sample):
    sample_scaled = scale(sample)
    return mdl.predict(sample_scaled)
