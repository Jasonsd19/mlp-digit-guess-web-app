import { useEffect, useRef, useState } from 'react'
import './App.css'
import axios from 'axios'

function App() {
  const canvas = useRef<HTMLCanvasElement | null>(null)
  const miniCanvas = useRef<HTMLCanvasElement | null>(null)

  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
  const [guess, setGuess] = useState<number>(-1)

  const [buttonTimeout, setButtonTimeout] = useState<number>(0);
  const [isWaiting, setIsWaiting] = useState<boolean>(false);

  const getCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvas?.current?.getBoundingClientRect()
    if (rect) {
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      return [x, y]
    }
    return [0, 0]
  }

  const setPosition = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const [x, y] = getCoords(e)
    setMousePos({ x: x, y: y })
  }

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.buttons !== 1) return

    if (canvas?.current) {
      const ctx = canvas.current.getContext('2d')
      if (ctx) {
        ctx.beginPath()
        ctx.moveTo(mousePos.x, mousePos.y);
        const [x, y] = getCoords(e);
        setMousePos({ x: x, y: y })
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    }
  }

  const updateMiniCanvas = () => {
    if (canvas?.current && miniCanvas?.current) {
      const ctx = canvas.current?.getContext('2d')
      const miniCtx = miniCanvas.current?.getContext('2d')
      if (ctx && miniCtx) {
        miniCtx.clearRect(0, 0, 28, 28)
        miniCtx.drawImage(canvas.current, 0, 0, miniCanvas.current.width, miniCanvas.current.height)
      }
    }
  }

  const timeoutSubmit = (timeout: number) => {
    if (timeout === 0) {
      setButtonTimeout(timeout)
      return
    }

    setButtonTimeout(timeout)
    setTimeout(() => {
      timeoutSubmit(timeout - 1)
    }, 1000)
  }

  const exportImage = () => {
    if (miniCanvas?.current) {
      const miniCtx = miniCanvas.current?.getContext('2d')
      if (miniCtx) {
        const imgData = miniCtx.getImageData(0, 0, 28, 28).data
        const normalizedData: number[] = []
        for (let i = 0; i < imgData.length; i += 4) {
          normalizedData.push(imgData[i + 3] * (1.0 / 255.0))
        }

        const config = {
          method: 'post',
          maxBodyLength: Infinity,
          url: 'Neural-Net-Prediction-Endpoint',
          headers: {
            'Content-Type': 'application/json'
          },
          data: normalizedData
        };

        setIsWaiting(true)
        axios.request(config).then((response) => {
          const data = response?.data
          setTimeout(() => {
            setGuess(parseInt(data))
            setIsWaiting(false)
          }, 250)
        }).catch((err) => console.log(err))
        timeoutSubmit(5)
      }
    }
  }

  const clear = () => {
    if (canvas?.current && miniCanvas?.current) {
      const ctx = canvas.current?.getContext('2d')
      const miniCtx = miniCanvas.current?.getContext('2d')
      if (ctx && miniCtx) {
        ctx.clearRect(0, 0, canvas.current.width, canvas.current.height)
        miniCtx.clearRect(0, 0, miniCanvas.current.width, miniCanvas.current.height)
      }
    }
  }

  const getGuessText = () => {
    if (isWaiting) return "Waiting for prediction..."
    if (guess === -1) return "Draw a number!"
    return guess
  }

  // Initialize line style for drawing on canvas
  useEffect(() => {
    if (canvas?.current && miniCanvas?.current) {
      const ctx = canvas.current.getContext('2d')
      if (ctx) {
        ctx.lineWidth = 19;
        ctx.lineCap = 'round';
        ctx.strokeStyle = '#000000';
      }
    }
  }, [])

  return (
    <div className="mainContainer">
      <div className='warning'>
        {"Sorry this web app is only supported on desktops with a minimum screen width of 1000 pixels."}
      </div>
      <div className='linkContainer'>
        <a href="https://github.com/Jasonsd19/immutable-matrices" target="_blank" rel="noreferrer noopener"> Matrix Library Code</a>
        <a href="https://github.com/Jasonsd19/multilayer-perceptron" target="_blank" rel="noreferrer noopener"> Multilayer Perceptron Code</a>
        <a href="https://www.jasondeol.com/" target="_blank" rel="noreferrer noopener">My Website</a>
        <a href="https://github.com/Jasonsd19/mlp-digit-guess-web-app" target="_blank" rel="noreferrer noopener">This Web App's Code</a>
      </div>
      <div className='headerContainer'>
        <p>Neural Network Digit Recognition</p>
      </div>
      <div className='neuralNetworkContainer'>
        <div className='smallDescriptonContainer'>
          <p>This is a neural network, specifically a multilayer perceptron (MLP), that I created entirely from scratch and trained to recognize the digits 0-9. This network was trained on the MNIST dataset, which is a set of handwritten digits.<br /><br /> This MLP in particular scored an accuracy of 97% on the test set. However there are still some limitations, because the MNIST dataset is standardized and most of the digits are centered on a 28x28 pixel grid, if you draw digits that aren't exactly centered on the image the network has trouble identifying it correctly.<br /><br /> All-in-all, I'm happy with the result and I learned a lot about machine learning and neural networks along the way. If you would like to know more feel free to read below, otherwise go ahead and test out the neural network!</p>
        </div>
        <div className='drawContainer'>
          <div className='mainCanvasContainer'>
            <canvas className='drawPad' width={448} height={448} onMouseDown={(e) => setPosition(e)} onMouseMove={(e) => draw(e)} onMouseUp={() => updateMiniCanvas()} onMouseLeave={() => updateMiniCanvas()} ref={canvas} />
          </div>
          <div className='miniCanvasContainer'>
            <canvas className='miniDrawPad' width={28} height={28} ref={miniCanvas} />
          </div>
          <div className='buttonContainer'>
            <button className='submitButton' onClick={() => clear()}>Clear</button>
            <button className='submitButton' onClick={() => exportImage()} disabled={buttonTimeout > 0}>{buttonTimeout > 0 ? `${buttonTimeout}...` : "Submit"}</button>
          </div>
        </div>
        <div className='neuralNetworkGuessContainer'>
          <div className='guessHeader'>
            <p>Neural Network Guess<br /><p style={{ fontSize: "13px" }}>(The first prediction takes ~30s)</p></p>
          </div>
          <div className='guess' style={{ fontSize: guess === -1 || isWaiting ? '30px' : '150px' }}>
            {getGuessText()}
          </div>
        </div>
      </div>
      <div className='exposition'>
        <p>This is a simple web application I developed. The neural network was implemented from scratch, and only has one dependency which was a linear algebra library I also created from scratch. As you draw on the canvas the mini-canvas is also updated to contain a scaled down version of whatever you draw. That tiny canvas is what is eventually sent to the neural network to be evaluated.<br /><br /> The reason it is done like this is because the neural network is trained on the MNIST dataset which contains hand-drawn digits that have been normalized to a 28x28 pixel image, the mini-canvas you see is also 28x28 pixels. Once the submit button is hit, the array of values that make up the 28x28 canvas is flattened to an array of 784 gray-scaled values ranging between 0 and 1. This is sent to a C++ based web server that parses the request and feeds it into the neural network, the prediction from the neural network is then parsed and sent back as a response.<br /><br /> I've already commented on how much I've learned about machine learning and neural networks, and how much intuition and understanding I've gained about the underlying principles of forward and back propagation, loss functions, and gradient descent. I also want to mention how much I've learned about coding and managing projects in C++, setting up web-servers, handling CORS headers and setting up API endpoints. This has definitely been a full-stack project and I've learned a ton along the way. If you're interested in any of the code involved in this project click the links at the top!</p>
      </div>
      <div className='footerContainer'>

      </div>
    </div >
  )
}

export default App
