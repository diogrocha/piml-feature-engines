MEMORY {
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 256K
  RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 16K
}
_estack = ORIGIN(RAM) + LENGTH(RAM);
SECTIONS {
  .vectors : { KEEP(*(.vectors)) } > FLASH
  .text : { *(.text*) *(.rodata*) } > FLASH
  _sidata = LOADADDR(.data);
  .data : {
    _sdata = . ;
    *(.data*)
    _edata = . ;
  } > RAM AT> FLASH
  .bss : {
    _sbss = . ;
    *(.bss*)
    *(COMMON)
    _ebss = . ;
  } > RAM
}
