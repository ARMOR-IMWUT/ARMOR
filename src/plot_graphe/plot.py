import matplotlib.pyplot as plt


class Arguments():
    def __init__(self):
        self.epochs = 110
        self.legends = ["Main task - Black box", "Main task - Model Replacement", "Attack success rate - black box",
                        "Attack success rate - model replacement"]
        self.maxy = 110
        self.ylabel = 'Accuracy (%)'
        self.xlabel = 'FL rounds'
        self.colors = ['black',
                       '#339933',
                       '#fdae61',
                       '#d7191c', '#000000', '#B03A2E', '#85C1E9']
        self.patterns = ['x', 'o', 'd', 'v', '^', '<', '>']


args = Arguments()
fig, ax = plt.subplots()
# data = [main_task_black_box, main_task_model_replacement, attack_model_black_box, attack_model_model_replacement]
data = []
for i in range(0, 4):

    if i == 0:
        ax.plot([0] + data[i], color=args.colors[i], marker=args.patterns[i], fillstyle='full', markevery=10)
    else:
        if i <= 3:
            ax.plot([0] + data[i], color=args.colors[i], marker=args.patterns[i], fillstyle='full', markevery=i * 5 + 3)
        else:
            ax.plot([0] + data[i], color=args.colors[i], marker=args.patterns[i - 3], fillstyle='full', markevery=i * 2)

plt.ylabel(args.ylabel)
plt.xlabel(args.xlabel)
plt.subplots_adjust(top=0.75)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=[0.5, 1.4])
ax.set_xlim(0, args.epochs)
ax.set_ylim(0, args.maxy)
plt.savefig("./attack_ndc_mnist.pdf", dpi=300)

plt.show()
