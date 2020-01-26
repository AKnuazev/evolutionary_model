from random import randint, random


class Individ:
    def __init__(self, intel=0, attract=0, age=0, MAX_AGE=100):
        self.intel = intel
        self.attract = attract
        self.age = age
        self.MAX_AGE = MAX_AGE

    # Mutation
    def mutate(self, probability=20):
        if randint(0, 99) < probability:
            if randint(0, 1):
                self.intel += random(-0.5, 0.5)
            else:
                self.attract += random(-0.5, 0.5)


    # Updates individ intelligenst
    # def study(self):
