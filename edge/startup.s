.syntax unified
.cpu cortex-m0
.thumb
.section .vectors,"a"
.word _estack
.word Reset_Handler
.text
.thumb_func
.global Reset_Handler
Reset_Handler:
  ldr r0,=_sdata
  ldr r1,=_edata
  ldr r2,=_sidata
1: cmp r0,r1
  bge 2f
  ldr r3,[r2]
  str r3,[r0]
  adds r0,r0,#4
  adds r2,r2,#4
  b 1b
2:
  ldr r0,=_sbss
  ldr r1,=_ebss
  movs r3,#0
3: cmp r0,r1
  bge 4f
  str r3,[r0]
  adds r0,r0,#4
  b 3b
4:
  bl main
5: b 5b
