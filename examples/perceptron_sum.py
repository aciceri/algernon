from algernon.perceptron import Perceptron
from random import uniform


def main():
    p = Perceptron(3)
    p.neuron.transfer = lambda x: x

    in_set = [(uniform(0, 1), uniform(0, 1), uniform(0, 1)) for _ in range(50)]
    out_set = [sum(n) for n in in_set]

    print("---START TRAINING---")
    p.train(in_set, out_set, max_epoch=5000, des_error=0.001, learning_rate=0.01)

    print("---TEST---")
    while True:
        try:
            a = float(input("First number: "))
            b = float(input("Second number: "))
            c = float(input("Third number: "))
            print("Approximate sum: %f" % p.go([a, b, c]))
            print("Desidered: %f" % (a + b + c))

        except KeyboardInterrupt:
            print("\nBye ;)")
            break

if __name__ == '__main__':
    main()
