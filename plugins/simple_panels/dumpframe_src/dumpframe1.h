#ifndef _PROCESSIMAGESEND2PANELS_H_
#define _PROCESSIMAGESEND2PANELS_H_

#include <usb.h>
#include <stdio.h>
#include <stdint.h>

typedef struct {
  int dummy;
} dumpframe_module_t;

typedef struct {
  dumpframe_module_t* module;
  usb_dev_handle *dev;
} dumpframe_device_t;

typedef enum {
  DUMPFRAME_SUCCESS = 0,
  DUMPFRAME_FAILURE = -1,
  DUMPFRAME_USB_DEVICE_NOT_FOUND = -2,
  DUMPFRAME_USB_SET_CONFIG_FAILED = -3,
  DUMPFRAME_USB_CLAIM_INTERFACE_FAILED = -4,
  DUMPFRAME_USB_BULK_WRITE_FAILED = -5,
  DUMPFRAME_MEMORY_ERROR = -6
} dumpframe_error_t;

/* Function Prototypes: */
/*#if defined(INCLUDE_FROM_PROCESSIMAGESEND2PANELS_C)*/
#define DUMPFRAME_API extern

DUMPFRAME_API dumpframe_error_t dumpframe_init(dumpframe_module_t**);
DUMPFRAME_API dumpframe_error_t dumpframe_close(dumpframe_module_t*);
DUMPFRAME_API dumpframe_error_t dumpframe_device_init(dumpframe_module_t* ,dumpframe_device_t**);
DUMPFRAME_API dumpframe_error_t dumpframe_device_close(dumpframe_device_t*);

DUMPFRAME_API dumpframe_error_t display_frame( dumpframe_device_t* device,
                                               void * data, intptr_t stride0,
                                               intptr_t shape0, intptr_t shape1,
                                               intptr_t offset0, intptr_t offset1 );
DUMPFRAME_API void say_hello();
/*#endif*/
#endif
