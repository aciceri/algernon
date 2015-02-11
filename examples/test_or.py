from algernon.perceptron import Perceptron


def main():
    p = Perceptron(2)

    in_set = [(0, 0), (0, 1), (1, 0), (1, 1)]
    out_set = [0, 1, 1, 1]

    print("---START TRAINING---")
    p.train(in_set, out_set, max_epoch=500, des_error=0, learning_rate=0.1)

    print("---TEST---")
    for x in (0, 1):
        for y in (0, 1):
            print("%d OR %d = %d" % (x, y, p.go([x, y])))

if __name__ == '__main__':
    main()
