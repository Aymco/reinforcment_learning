
from ai import AI_MODEL
from car_game import Game


if __name__ == '__main__':



    # train the model
    ai = AI_MODEL(train=True, layers=[7, 4, 3, 2], n_variations=100)

    ai_name = 'model_1'
    circuit_name = 'circuit_1'

    #ai.load(ai_name)
    #ai.REMAKE_CIRCUIT = True
    ai.init()

    ai.create_variations()
    cars = ai.get_cars()
    game = Game(cars=cars , show_objects=False, circuit=circuit_name, screen=True)
    game.show_screen = True
    game.game_loop()

    # save the model
    ai.save(ai_name)    

    # save the circuit
    game.Circuit.save(circuit_name)
    