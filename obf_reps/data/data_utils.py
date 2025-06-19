from obf_reps.data import ConceptDataset, ObfusDataset


def join_concept_obfus_datasets(
    concept_data: ConceptDataset, obfus_data: ObfusDataset, *, is_negative_obfus: bool
) -> ConceptDataset:
    """Join concept and obfuscation datasets.

    Concept is list of tuples of pos and neg (prompt, statement) Obfus is list of tuples (input,
    behavior_target, rep_source)
    """
    assert is_negative_obfus is False, "Only negative obfuscation is supported"

    concept_size = len(concept_data)
    obfus_size = len(obfus_data)

    pos_concept_data = [tup[0] for tup in concept_data]
    neg_concept_data = [tup[1] for tup in concept_data]

    # Whether the concept data has a target
    add_target = pos_concept_data[0][1] is not None

    if is_negative_obfus:
        new_concept_data = [
            (
                pos_concept_data[i % concept_size],
                (obfus_data[i][0], obfus_data[i][1] if add_target else None),
            )
            for i in range(obfus_size)
        ]
    else:
        new_concept_data = [
            (
                (obfus_data[i][0], obfus_data[i][1] if add_target else None),
                neg_concept_data[i % concept_size],
            )
            for i in range(obfus_size)
        ]

    return concept_data + new_concept_data
