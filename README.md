# mlp-digit-guess-web-app
This repository is not meant to used for deployment or anything like that, I just wanted to have both my front-end and back-end files in one convenient place. This makes it easier for anyone interested in the code to take a look and browse the source. I do however have two separate private repositories that I use for actually deploying the full web application, which you can find [here](https://perceptron.jasondeol.com/). Anyways, onto the write up...

## Front-End
The front end of this web application was created using React, TypeScript, HTML, and CSS. I didn't spend too much styling or making the website completely responsive. I mainly focused on making sure that the drawing canvas was implemented really well, and that the calls to the backend POST endpoint worked as intended. It was a fun little project and I still learned a decent amount. If you want to see a better representation of my designing abilities check out my [personal website](https://www.jasondeol.com/) :grin:

## Back-End
The back-end is where most of the magic happens, and where most of the work went into. I already had an implementation of the multilayer perceptron, and the matrix library that it depended on, check those out [here](https://github.com/Jasonsd19/multilayer-perceptron) and [here](https://github.com/Jasonsd19/immutable-matrices). With the implementation for a basic neural network in hand and the MNIST hand-drawn digit dataset I got to work. 

The first course of action was creating a simple program to parse the dataset CSV files, this implementation can be found under /csvParser. Next I had to implement a method of reading/writing a models data to/from a file, this is very useful when training a model as you can just store the weights and biases and retrieve them later. Lastly, I had to set-up the actual webServer that would take in the POST requests and send back a response containing the neural networks prediction. For this I utilized the only external dependecies [uSockets](https://github.com/uNetworking/uSockets) + [uWebSockets](https://github.com/uNetworking/uWebSockets) and [JSON](https://github.com/nlohmann/json).

## How it works
The user draws on a canvas on the webpage, everytime they draw on the large canvas the small 28x28 pixel canvas below also gets updated, creating a down-sized version of the original drawing. Since the MNIST dataset contains pictures standardized to a 28x28 pixel array this is perfect. Once the submit button is hit by the user the data in the pixel array of the small canvas is normalized to be between 0 and 1, and the 1D canvas containing 784 entries is sent to the back-end via a POST request. The back-end receives and parses the request body to retrieve the data, the data is then passed into the pre-trained model which gives back a prediction. The prediction is parsed and sent back to the webpage where the viewer can see it. The webpage is deployed using its static build files, and the back-end is running on a docker container.

## Changes
Looking back at this project, the major issue that stuck out to me was training time. I believe the current trained model that I use on the live web app took ~2 days to train. Since I used a form of stochastic gradient descent the weights were updated after each training example, this made it hard to parallelize and so I had the network training on a single core. Ideally, when using large datasets I should have implemented a type of batch gradient descent which would have allowed me to train the network in parallel and would have significantly reduced the amount of time it took to train.

## In conclusion...
I learned a lot. Building this web application from the ground-up was a very fun and challenging experience and taught me a lot about not only machine learning, but also full-stack development. From creating a website, to building and training a neural network, setting up API endpoints, and having to manage two separate deployments, it was an interesting journey. If you would like to test out the neural network youself check it out [here](https://perceptron.jasondeol.com/).
