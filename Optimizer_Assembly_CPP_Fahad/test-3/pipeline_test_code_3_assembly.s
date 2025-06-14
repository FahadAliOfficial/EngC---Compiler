.global main
.text

main:
    # Function prologue
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp

    # return 78.5

    # Function epilogue
    movq %rbp, %rsp
    popq %rbp
    ret