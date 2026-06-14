use std::fs;

fn read_count(path: &str) -> i32 {
    let contents = fs::read_to_string(path).unwrap();
    contents.trim().parse::<i32>().unwrap()
}

fn first_word(s: &str) -> &str {
    s.split(' ').next().unwrap()
}
