
from ai import AI_MODEL
from car_game import Game


from mem_nn import MemNN
from pytorch_nn import torchNN


if __name__ == '__main__':

    TRAIN = True


    # train the model
    
    ai = AI_MODEL(train=TRAIN, layers=[4, 4, 2], n_variations=500, n_duplicates=1, nn=torchNN)

    ai_name = 'model_all'
    circuit_name = 'circuit_1'

    #ai.load(ai_name)
    ai.nn.init()
    ai.REMAKE_CIRCUIT = True
    
    ai.create_variations()
    cars = ai.get_cars()
    game = Game(cars=cars , show_objects=False, circuit=circuit_name, screen=True, training=TRAIN)
    game.show_screen = True
    game.game_loop()

    # save the model
    ai.save(ai_name)

    # save the circuit
    circuit_name = 'circuit_1'
    game.Circuit.save(circuit_name)
    