import matplotlib.pyplot as plt

MODEL_TOP_5_PERFORMANCE = {
    "AlexNet":15.3,
    "VGG16": 7.3,
    "Humans": 5.1,
    "ResNet": 3.57,
    "EfficientNet":2.9,
}

YEARS = [2012,2013,2014,2015,2019]
TOP_5_ERROR_RATE = [16.4,11.7,7.3,3.57,2.9]
MODEL_NAMES = ["AlexNet","ZFNet","VGG16","ResNet","EfficientNet"]

def create_plot():
    plt.figure(figsize=(10,6))
    bars = plt.bar(YEARS,TOP_5_ERROR_RATE, color='skyblue')

    # add model names, with their error rate on top of the bars
    for bar, model_name, error_rate in zip(bars, MODEL_NAMES, TOP_5_ERROR_RATE):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.5, f"{model_name} ({error_rate}%)", ha='center', va='top')

    # add a horionztal dotted linme at y=5.1
    plt.axhline(y=5.1, color='r', linestyle='--')
    # label it as human error rate with a text on the right, the text should be red
    plt.text(2019, 5.1, 'Human error rate (5.1%)', ha='right', va='top')


    plt.xlabel('Year')
    plt.ylabel('Top-5 Error Rate (%)')
    plt.title('Top-5 Error Rate of CNNs on ImageNet (from 2012-2019)')

    plt.show()

if __name__ == "__main__":
    create_plot()