from schrl.tltl.predicate import NeuralPredicate, ProgrammablePredicate
from schrl.tltl.spec import DiffTLTLSpec


def test_syntax():
    a = ProgrammablePredicate(lambda x: x < 10, "a")
    spec1 = DiffTLTLSpec(a)
    b = NeuralPredicate(None, "b")  # type: ignore
    spec2 = DiffTLTLSpec(b)

    print(spec1 & spec2)
    print(spec1 | spec2)
    print(~spec1 | spec2)
    print(~spec1 | spec2.eventually())
    print(~(spec1 | spec2))
    print(~spec1 | spec2 | spec1)
    print(~spec1 | spec2 & spec1)
    print(~(spec1 | spec2) & spec1)
    print(~(~(spec1 | spec2)))
    print(spec1.implies(spec2))
    print(spec1.next())
    print(~((spec1 | spec2).until(spec2)))
    print(~((spec1 | spec2).eventually()))
    print(~(spec1 | spec2).globally())
    print((~(spec1 | spec2)).globally())
    print(spec1.globally().until(spec2))
    print(spec1.until(spec2).globally())


if __name__ == '__main__':
    test_syntax()
