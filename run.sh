if [ ! -d "archive.physionet.org/pn6/chbmit/" ]; then
    echo "Downloading data"
    wget -r --no-parent https://archive.physionet.org/pn6/chbmit/
    echo "Downloaded data"
fi
echo "Creation of the spectograms"
echo ""
python3 DatasetToSpectogram.py
echo "Creation of the CNN and evaluation of the model on the spectograms"
echo ""
python3 CNN.py
echo "Search the best thresold for each patient"
echo ""
python3 TestThreshold.py
echo ""
echo "finished"