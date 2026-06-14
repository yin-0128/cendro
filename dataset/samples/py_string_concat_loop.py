def render_rows(rows):
    html = ""
    for row in rows:
        html += "<tr>"
        for cell in row:
            html += "<td>" + str(cell) + "</td>"
        html += "</tr>"
    return html


def join_paths(parts):
    result = ""
    for p in parts:
        result = result + "/" + p
    return result
