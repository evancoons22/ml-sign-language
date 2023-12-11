# ml-sign-language
Live inference of sign language letters and digits for math 156 final project UCLA.  
Built with pytorch and [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).

### Example

<img src="https://github.com/evancoons22/ml-sign-language/blob/main/output2.gif" width="640">


### Instructions
1. Make sure these dependencies are installed:
   - `$pip install flask`
   - `$pip install flask-cors`
   - `$pip install torch-geometric`
   - install torch, numpy
2. Run flask with `$python app.py`
3. Open index.html in browser.
4. Turn off CORS in the browser (in Safari: Developer > Disable Cross-Origin Restrictions) 

Can add to training dataset with label and send data button. 
