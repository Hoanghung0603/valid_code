from medret.index.embed_input import passage_input, query_input


def test_first_chunk_already_starts_with_title():
    text = "Bị nứt kẽ hậu môn bôi thuốc gì?\nNứt hậu môn là tình trạng rách ở niêm mạc hậu môn."
    assert passage_input("Bị nứt kẽ hậu môn bôi thuốc gì?", "", text) == text


def test_title_and_outer_headings_prefix_a_later_chunk():
    text = "Thuốc bôi giúp giảm đau và làm lành vết nứt."
    got = passage_input("Bị nứt kẽ hậu môn bôi thuốc gì?", "3. Điều trị > 3.1 Thuốc bôi", text)
    assert got == "Bị nứt kẽ hậu môn bôi thuốc gì? 3. Điều trị > 3.1 Thuốc bôi " + text


def test_heading_the_chunk_starts_with_is_not_repeated():
    text = "3.1 Thuốc bôi\nThuốc bôi giúp giảm đau."
    got = passage_input("Nứt kẽ hậu môn", "3. Điều trị > 3.1 Thuốc bôi", text)
    assert got == "Nứt kẽ hậu môn 3. Điều trị " + text


def test_query_has_no_instruction():
    assert query_input("  Bé  trai 3 tháng\nbị giang mai? ") == "Bé trai 3 tháng bị giang mai?"
