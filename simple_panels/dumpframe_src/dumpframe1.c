/*
	USBtoPanels Board: USB Host Source Code 

    who when        what
    --- ----        ----
    pjp 04/20/08    version 1.0
*/

#define INCLUDE_FROM_PROCESSIMAGESEND2PANELS_C
#include "dumpframe1.h"

/* Global Variables: */
/* Dangerous, but easier to access across functions */
USBPacketArrayWrapper_t USBPacketArray;
PanelArrayWrapper_t     PanelArray;
FrameImageWrapper_t     FrameImage;

int display_frame( void * data, intp stride0, intp shape0, intp shape1, intp offset0, intp offset1 )
{    
    /* Convert input image to size and shape necessary to display on panels */
    InputImage2FrameImage(data, stride0, shape0, shape1, offset0, offset1 );
    
    /* Determine addresses and ImageMatrix starting indicies for each panel displaying FrameImage, store values in PanelArray */
    MapFrameImage2PanelArray();
    
    /* Convert FrameImage data to panel message for each panel in PanelArray */
    ConvertFrameImage2PanelMessages();
    
    /* Move PanelArray data into USBPacketArray to prepare for usb_bulk_write */
    PanelArray2USBPacketArray();
    
    /* usb_bulk_write every usb packet in USBPacketArray */
    USBBulkWriteUSBPacketArray();
}

static void InputImage2FrameImage( void * data, intp stride0, intp shape0, intp shape1, intp offset0, intp offset1 )
{
    double inRowPos,inColPos;
    unsigned short fiRowN,fiColN,fiRowNum,fiColNum,fiColOver,fiColStuff=0,fiColOffset=0;
    fiRowNum = FRAME_PANEL_ROW_NUM*PANEL_PIXEL_NUM; /*Scale display image to span top to bottom of frame*/
    /* TO DO: Write code to account for images smaller than frame */
    double scaleFactor = shape0/fiRowNum;
    fiColNum = (shape1*fiRowNum)/shape0;  /*1:1 aspect ratio*/
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
    /* Sub-sample input image and store values into FrameImage.Matrix in appropriate locations*/
    for(fiRowN=0; fiRowN<fiRowNum; fiRowN++) {
        inRowPos = (fiRowN+0.5)*scaleFactor;
        for(fiColN=0; fiColN<fiColNum; fiColN++) {
            inColPos = (fiColN+0.5)*scaleFactor;
            FrameImage.ImageMatrix[fiRowN][fiColN+fiColOffset] = *(unsigned char*)(data+(unsigned short)inRowPos*stride0+(unsigned short)inColPos);
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
    unsigned char rowValArray[3],colValArray[3],rowValArrayComp[3];
    unsigned char GrayScaleFactor = 256/(0x01 << PANEL_GRAY_SCALE_BITS);
    unsigned char panelProcessed,compressible; 
    
    for (panelN=0; panelN<PanelArray.PanelsInThisArray; panelN++) {
        piRowOffset = PanelArray.Panel[panelN].FrameImageIndex.Row;
        piColOffset = PanelArray.Panel[panelN].FrameImageIndex.Col;

        panelProcessed = 0;
        /* Check to see if all the rows in the panel image are identical */ 
        compressible = 1;
        piRowN = 0;
        while (compressible && (piRowN<PANEL_PIXEL_NUM)) {
            rowValArray[0] = 0;
            rowValArray[1] = 0;
            rowValArray[2] = 0;
            for (piColN=0; piColN<PANEL_PIXEL_NUM; piColN++) {
                pixelVal = FrameImage.ImageMatrix[piRowOffset+piRowN][piColOffset+piColN];
                pixelVal /= GrayScaleFactor;
                rowValArray[0] |=  (pixelVal & 0x01)       << piColN;
                rowValArray[1] |= ((pixelVal & 0x02) >> 1) << piColN;
                rowValArray[2] |= ((pixelVal & 0x04) >> 2) << piColN;
            }
            if (piRowN==0) {
                rowValArrayComp[0] = rowValArray[0];
                rowValArrayComp[1] = rowValArray[1];
                rowValArrayComp[2] = rowValArray[2];
            } else {
                if ((rowValArrayComp[0]!=rowValArray[0])||(rowValArrayComp[1]!=rowValArray[1])||(rowValArrayComp[2]!=rowValArray[2])) {
                    compressible = 0;
                }
            }
            piRowN++;
        }
        /* If the rows in the panel image are identical, row or identity compress the data */
        if (compressible) {
            if ((rowValArray[0]==rowValArray[1])&&(rowValArray[0]==rowValArray[2])) {
                /* Identity compression */
                PanelArray.Panel[panelN].Message[0] = rowValArray[0];
                PanelArray.Panel[panelN].Header.BytesInPanelMessage = 1;
                PanelArray.Panel[panelN].MessageIndex = 0;
            } else {
                /* Row compression */
                PanelArray.Panel[panelN].Message[0] = rowValArray[2];
                PanelArray.Panel[panelN].Message[1] = rowValArray[1];
                PanelArray.Panel[panelN].Message[2] = rowValArray[0];
                PanelArray.Panel[panelN].Header.BytesInPanelMessage = PANEL_GRAY_SCALE_BITS;
                PanelArray.Panel[panelN].MessageIndex = 3 - PANEL_GRAY_SCALE_BITS;
            }
            panelProcessed = 1;
        }
        
        /* If panel is incompressible, proceed to full message conversion */
        if (!panelProcessed) {
            for (piColN=0; piColN<PANEL_PIXEL_NUM; piColN++) {
                colValArray[0] = 0;
                colValArray[1] = 0;
                colValArray[2] = 0;
                for (piRowN=0; piRowN<PANEL_PIXEL_NUM; piRowN++) {
                    pixelVal = FrameImage.ImageMatrix[piRowOffset+piRowN][piColOffset+piColN];
                    pixelVal /= GrayScaleFactor;
                    colValArray[0] |=  (pixelVal & 0x01)       << piRowN;
                    colValArray[1] |= ((pixelVal & 0x02) >> 1) << piRowN;
                    colValArray[2] |= ((pixelVal & 0x04) >> 2) << piRowN;
                }
                PanelArray.Panel[panelN].Message[piColN]                    = colValArray[2];
                PanelArray.Panel[panelN].Message[piColN+PANEL_PIXEL_NUM]    = colValArray[1];
                PanelArray.Panel[panelN].Message[piColN+PANEL_PIXEL_NUM*2]  = colValArray[0];
            }
            PanelArray.Panel[panelN].Header.BytesInPanelMessage = PANEL_GRAY_SCALE_BITS*PANEL_PIXEL_NUM;
            PanelArray.Panel[panelN].MessageIndex = PANEL_MESSAGE_BUFFER_SIZE - PANEL_GRAY_SCALE_BITS*PANEL_PIXEL_NUM;
        }
    }
}

static void PanelArray2USBPacketArray(void)
{
   unsigned char packetN=0,panelN=0,panelsInPacket,packetByteN,messageByteN;

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
        printf("Error: Device not found!\n");
        return;
    }

    if(usb_set_configuration(dev, 1) < 0)
    {
        printf("Error: Setting config 1 failed\n");
        usb_close(dev);
        return;
    }

    if(usb_claim_interface(dev, 0) < 0)
    {
        printf("Error: Claiming interface 0 failed\n");
        usb_close(dev);
        return;
    }
    
    for (packetN = 0; packetN<USBPacketArray.PacketsInThisArray; packetN++) {
        outBufSize = USBPacketArray.Packet[packetN].BytesInPacketData;
        bytesWritten = usb_bulk_write(dev, USB_EP_OUT_NUM, USBPacketArray.Packet[packetN].Data, outBufSize, 5000);
        if(bytesWritten != outBufSize)
        {
            printf("Error: Bulk write failed\n");
        }
    }
    /*
    bytesRead = usb_bulk_read(dev, USB_EP_OUT_NUM, in_tmp, sizeof(in_tmp), 5000);
    if(bytesRead != sizeof(in_tmp))
    {
        printf("Error: Bulk read failed\n");
    }
    */

    usb_release_interface(dev, 0);
    usb_close(dev);

    return;
}

void say_hello() {
  printf("hello!!\n");
}

