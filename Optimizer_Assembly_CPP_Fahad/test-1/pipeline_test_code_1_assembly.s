.global main
.text

main:
    # Function prologue
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp

    # return 15
    movq $15, %rax

    # Function epilogue
    movq %rbp, %rsp
    popq %rbp
    ret