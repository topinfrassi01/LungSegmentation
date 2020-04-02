// Labels are in the GIF format with a 1024x1024 dimension.
// This script takes every GIF in a folder and converts them to 256x256 PNG images.

setBatchMode(true);
dir = getDirectory("Choose a Directory ");
list = getFileList(dir);
for (i=0; i<list.length; i++) {
    if (endsWith(list[i], ".gif")) {
        open(list[i]);
        run("Scale...", "x=- y=- width=256 height=256 interpolation=Bilinear average create title=window");
        selectWindow("window");
        saveAs("PNG", dir + list[i] + ".png");

        while (nImages>0) { 
            selectImage(nImages); 
            close(); 
        } 
    }
}
