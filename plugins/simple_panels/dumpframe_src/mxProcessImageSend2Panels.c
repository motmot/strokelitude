/*=================================================================
 * mexfunction.c
 *
 * tic;[OutputImage]=mxProcessImageSend2Panels(InputImage);z=toc;
 * mex mxProcessImageSend2Panels.c -lusb
 * 
=================================================================*/

/*
	USB to panels application. 

    who when        what
    --- ----        ----
    pjp 04/08/08    version 1.0
*/

#define INCLUDE_FROM_PROCESSIMAGESEND2PANELS_C
#include "mxProcessImageSend2Panels.h"

/* Global Variables: */
/* Dangerous, but easier to access in functions */
USBPacketArrayWrapper_t USBPacketArray;
PanelArrayWrapper_t     PanelArray;
FrameImageWrapper_t     FrameImage;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    unsigned char dimNum;
    mwSize inputImageDimArray[2],outputImageDimArray[2];
    unsigned char *pInputImage,*pOutputImage;
    
    /* Check for proper number of input and output arguments */    
    if (nrhs != 1) {
        mexErrMsgTxt("One input argument required.");
    } 
    if (nlhs > 0){
        mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument */
    if (!(mxIsUint8(MEXINPUT))) {
        mexErrMsgTxt("Input array must be of type uint8.");
    }
    
    /* Check dimension number of input argument */
    dimNum = (unsigned char)mxGetNumberOfDimensions(MEXINPUT);
    if (dimNum != 2) {
        mexErrMsgTxt("Input argument dimensions must equal 2.");
    }
    
    /* Get input image, along with its size info */
    inputImageDimArray[0] = mxGetM(MEXINPUT); 
    inputImageDimArray[1] = mxGetN(MEXINPUT);
    pInputImage = (unsigned char *)mxGetPr(MEXINPUT);
    
    /* Convert input image to size and shape necessary to display on panels */
    InputImage2FrameImage(pInputImage, inputImageDimArray);
    
    /* Determine addresses and ImageMatrix starting indicies for each panel displaying FrameImage, store values in PanelArray */
    MapFrameImage2PanelArray();
    
    /* Convert FrameImage data to panel message for each panel in PanelArray */
    ConvertFrameImage2PanelMessages();
    
    /* Move PanelArray data into USBPacketArray to prepare for usb_bulk_write */
    PanelArray2USBPacketArray();
    
    /* usb_bulk_write every usb packet in USBPacketArray */
    USBBulkWriteUSBPacketArray();
    
    /*
    outputImageDimArray[0] = (mwSize)FrameImage.RowNum;
    outputImageDimArray[1] = (mwSize)FrameImage.ColNum;
    MEXOUTPUT = mxCreateNumericArray(dimNum,outputImageDimArray,mxUINT8_CLASS,mxREAL);
    pOutputImage = (unsigned char *)mxGetPr(MEXOUTPUT); 
    
    FrameImage2OutputImage(pOutputImage, outputImageDimArray);
    */
}

static void InputImage2FrameImage(unsigned char inArray[], mwSize inDimArray[])
{
    double inRowPos,inColPos;
    unsigned short fiRowN,fiColN,fiRowNum,fiColNum,fiColOver,fiColStuff=0,fiColOffset=0;
    fiRowNum = FRAME_PANEL_ROW_NUM*PANEL_PIXEL_NUM; /*Scale display image to span top to bottom of frame*/
    /* TO DO: Write code to account for small images
    if (fiRowNum>inDimArray[0]) {
        fiRowNum = inDimArray[0];   Unless input image is smaller than frame
    }
    */
    double scaleFactor = inDimArray[0]/fiRowNum;
    fiColNum = (inDimArray[1]*fiRowNum)/inDimArray[0];  /*1:1 aspect ratio*/
    fiColOver = fiColNum % PANEL_PIXEL_NUM;
    /* Add pixel columns if necessary to center image within panels */
    if (fiColOver) {
        fiColStuff = PANEL_PIXEL_NUM - fiColOver;
    }
    /* Add a panel column if necessary to center image within frame */
    if ((FRAME_PANEL_COL_NUM-((fiColNum+fiColStuff)/PANEL_PIXEL_NUM))%2) {
        fiColStuff += PANEL_PIXEL_NUM;
    }
    if (fiColStuff) {
        fiColOffset = fiColStuff/2;
        /* No use sending an entire panel column of just zeros */
        if ((fiColStuff-fiColOffset)==PANEL_PIXEL_NUM) {
            fiColStuff -= PANEL_PIXEL_NUM;
        }
        /* Initialize all values in FrameImage.Matrix to 0, in case 'stuffed' regions contain garbage */
        for(fiColN=0; fiColN<(fiColNum+fiColStuff); fiColN++) {
            for(fiRowN=0; fiRowN<fiRowNum; fiRowN++) {
                FrameImage.ImageMatrix[fiRowN][fiColN] = 0;
            }
        }
    }
    /* Sample input image and store values into FrameImage.Matrix in appropriate locations*/
    for(fiColN=0; fiColN<fiColNum; fiColN++) {
        inColPos = (fiColN+0.5)*scaleFactor;
        for(fiRowN=0; fiRowN<fiRowNum; fiRowN++) {
            inRowPos = (fiRowN+0.5)*scaleFactor;
            FrameImage.ImageMatrix[fiRowN][fiColN+fiColOffset] = *(inArray+(unsigned short)inRowPos+(unsigned short)inColPos*inDimArray[0]);
        }
    }
    /* Update FrameImage size info */
    FrameImage.RowNum = fiRowNum;
    FrameImage.ColNum = fiColNum + fiColStuff;
}

static void MapFrameImage2PanelArray(void)
{
    unsigned char paRowN,paColN,paRowNum,paColNum,paColOffset,panelN=0;
    unsigned char fiRowStart,fiColStart;
    paRowNum = FrameImage.RowNum/PANEL_PIXEL_NUM;
    paColNum = FrameImage.ColNum/PANEL_PIXEL_NUM;
    paColOffset = (FRAME_PANEL_COL_NUM - paColNum)/2;
    
    for (paRowN=0; paRowN<paRowNum; paRowN++) {
        fiRowStart = paRowN*PANEL_PIXEL_NUM;
        for (paColN=0; paColN<paColNum; paColN++) {
            fiColStart = paColN*PANEL_PIXEL_NUM;
            PanelArray.Panel[panelN].Header.PanelAddress = PanelAddressMatrix[paRowN][paColN+paColOffset];
            PanelArray.Panel[panelN].FrameImageIndex.Row = fiRowStart;
            PanelArray.Panel[panelN].FrameImageIndex.Col = fiColStart;
            panelN++;
            PanelArray.PanelsInThisArray = panelN;
        }
    }
}

static void ConvertFrameImage2PanelMessages(void)
{
    unsigned char panelN,pixelVal,piRowN,piColN,piRowOffset,piColOffset;
    unsigned char colValArray[3];
    unsigned char GrayScaleFactor = 256/(0x01 << PANEL_GRAY_SCALE_BITS);
    
    for (panelN=0; panelN<PanelArray.PanelsInThisArray; panelN++) {
        piRowOffset = PanelArray.Panel[panelN].FrameImageIndex.Row;
        piColOffset = PanelArray.Panel[panelN].FrameImageIndex.Col;

        /*mexPrintf("panelAddress: %d, Message:\n",PanelArray.Panel[panelN].Header.PanelAddress);*/
        
        for (piColN=0; piColN<PANEL_PIXEL_NUM; piColN++) {
            colValArray[0] = 0;
            colValArray[1] = 0;
            colValArray[2] = 0;
            for (piRowN=0; piRowN<PANEL_PIXEL_NUM; piRowN++) {
                pixelVal = FrameImage.ImageMatrix[piRowOffset+piRowN][piColOffset+piColN];
                /*mexPrintf("%d,",pixelVal);*/
                pixelVal /= GrayScaleFactor;
                /*mexPrintf("%d\t",pixelVal);*/
                colValArray[0] |=  (pixelVal & 0x01)       << piRowN;
                colValArray[1] |= ((pixelVal & 0x02) >> 1) << piRowN;
                colValArray[2] |= ((pixelVal & 0x04) >> 2) << piRowN;
            }
            /*mexPrintf("\n");*/
            PanelArray.Panel[panelN].Message[piColN]                    = colValArray[2];
            PanelArray.Panel[panelN].Message[piColN+PANEL_PIXEL_NUM]    = colValArray[1];
            PanelArray.Panel[panelN].Message[piColN+PANEL_PIXEL_NUM*2]  = colValArray[0];
        }
        PanelArray.Panel[panelN].Header.BytesInPanelMessage = PANEL_GRAY_SCALE_BITS*PANEL_PIXEL_NUM;
        PanelArray.Panel[panelN].MessageIndex = PANEL_MESSAGE_BUFFER_SIZE - PANEL_GRAY_SCALE_BITS*PANEL_PIXEL_NUM;
        /*
        unsigned char temp;    
        for (temp=PanelArray.Panel[panelN].MessageIndex; temp<24; temp++) {
            mexPrintf("%d\t",PanelArray.Panel[panelN].Message[temp]);
        }
        mexPrintf("\n\n");
        */
    }
}

static void PanelArray2USBPacketArray(void)
{
   unsigned char packetN=0,panelN=0,panelsInPacket,packetByteN,messageByteN;
   
   /*mexPrintf("USBPackets: \n");*/
   
   while (panelN<PanelArray.PanelsInThisArray) {
       USBPacketArray.Packet[packetN].Data[0] = 0;          /*USBPacket.Header.ControlByte*/
       USBPacketArray.Packet[packetN].Data[1] = packetN;    /*USBPacket.Header.USBPacketIDNumber*/
       packetByteN = sizeof(USBPacketArray.Packet[packetN].Header);
       panelsInPacket = 0;
       while ((panelN<PanelArray.PanelsInThisArray)&&((USB_EP_OUT_SIZE-packetByteN)>(PanelArray.Panel[panelN].Header.BytesInPanelMessage+sizeof(PanelArray.Panel[panelN].Header)))) {
           USBPacketArray.Packet[packetN].Data[packetByteN++] = PanelArray.Panel[panelN].Header.PanelAddress;
           USBPacketArray.Packet[packetN].Data[packetByteN++] = PanelArray.Panel[panelN].Header.BytesInPanelMessage;
           for (messageByteN=0; messageByteN<PanelArray.Panel[panelN].Header.BytesInPanelMessage; messageByteN++) {
               USBPacketArray.Packet[packetN].Data[packetByteN++] = PanelArray.Panel[panelN].Message[PanelArray.Panel[panelN].MessageIndex+messageByteN];
           }
           panelsInPacket++;
           panelN++;
       }
       USBPacketArray.Packet[packetN].BytesInPacketData = packetByteN;
       USBPacketArray.Packet[packetN].Data[2] = panelsInPacket; /*USBPacket.Header.PanelsInThisUSBPacket*/
       packetN++;
       USBPacketArray.PacketsInThisArray = packetN;
   }
   /*
   unsigned char pN;
   unsigned char dN;
   for (pN=0; pN<USBPacketArray.PacketsInThisArray; pN++) {
       for (dN=0; dN<USBPacketArray.Packet[pN].BytesInPacketData; dN++) {
           mexPrintf("%d\t",USBPacketArray.Packet[pN].Data[dN]);
       }
       mexPrintf("\n");
   }
    */
}

static usb_dev_handle *OpenUSBDev(void)
{
    struct usb_bus *bus;
    struct usb_device *dev;

    for(bus = usb_get_busses(); bus; bus = bus->next) 
    {
        for(dev = bus->devices; dev; dev = dev->next) 
        {
            if(dev->descriptor.idVendor == USB_DEV_VID
                && dev->descriptor.idProduct == USB_DEV_PID)
            {
                return usb_open(dev);
            }
        }
    }
    return NULL;
}

static void USBBulkWriteUSBPacketArray(void)
{
    usb_dev_handle *dev = NULL;                 /* Device handle */
    unsigned char bytesWritten,packetN,outBufSize;

    usb_init();                                 /* Initialize USB library */
    usb_find_busses();                          /* Find all USB busses */
    usb_find_devices();                         /* Find all connected USB devices */

    if(!(dev = OpenUSBDev()))
    {
        mexWarnMsgTxt("Error: Device not found!\n");
        return;
    }

    if(usb_set_configuration(dev, 1) < 0)
    {
        mexWarnMsgTxt("Error: Setting config 1 failed\n");
        usb_close(dev);
        return;
    }

    if(usb_claim_interface(dev, 0) < 0)
    {
        mexWarnMsgTxt("Error: Claiming interface 0 failed\n");
        usb_close(dev);
        return;
    }
    
    for (packetN = 0; packetN<USBPacketArray.PacketsInThisArray; packetN++) {
        outBufSize = USBPacketArray.Packet[packetN].BytesInPacketData;
        bytesWritten = usb_bulk_write(dev, USB_EP_OUT_NUM, USBPacketArray.Packet[packetN].Data, outBufSize, 5000);
        if(bytesWritten != outBufSize)
        {
            mexWarnMsgTxt("Error: Bulk write failed\n");
        }
    }
    /*
    bytes_read = usb_bulk_read(dev, EP_IN, in_tmp, sizeof(in_tmp), 5000);
    if(bytes_read != sizeof(in_tmp))
    {
        printf("error: bulk read failed\n");
    }
    else
    {
        for (i = 0; i < sizeof(in_tmp); i++)
        {
            responseData[i] = in_tmp[i];
        }
    }
    */

    usb_release_interface(dev, 0);
    usb_close(dev);

    return;
}

static void FrameImage2OutputImage(unsigned char outArray[], mwSize outDimArray[])
{
    unsigned short outRowN,outColN;
       
    for(outColN=0; outColN<outDimArray[1]; outColN++) {
        for(outRowN=0; outRowN<outDimArray[0]; outRowN++) {
            *(outArray+outRowN+outColN*outDimArray[0]) = FrameImage.ImageMatrix[outRowN][outColN];
        }
    }
}
