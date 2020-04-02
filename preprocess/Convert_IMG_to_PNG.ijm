// Images from the Dataset are in the IMG Raw format according to specifications found in 

setBatchMode(true);
dir = getDirectory("Choose a Directory ");
list = getFileList(dir);
for (i=0; i<list.length; i++) {
    if (endsWith(list[i], ".IMG")) {
        run("Raw...", "open="+dir + list[i]+" image=[16-bit Unsigned] width=2048 height=2048 white");
	run("Scale...", "x=- y=- width=256 height=256 interpolation=Bilinear average create title=window");
	selectWindow("window");
	saveAs("PNG", dir + list[i] + ".png");

	while (nImages>0) { 
            selectImage(nImages); 
            close(); 
        } 
    }
}