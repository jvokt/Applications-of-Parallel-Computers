/^##/ { indoc = 1; next }
indoc && !/^#/ { indoc = 0 }

indoc { sub(/^#[ ]?/, "") }
!indoc { $0 = "    " $0 }

{ print $0 }
