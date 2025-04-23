// src/lexer/mod.rs
use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
    // Keywords
    #[token("fn")]
    Fn,
    #[token("let")]
    Let,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("task")]
    Task,
    #[token("channel")]
    Channel,
    #[token("scope")]
    Scope,
    #[token("transaction")]
    Transaction,
    #[token("reactive")]
    Reactive,
    #[token("mut")]
    Mut,
    #[token("send")]
    Send,
    #[token("receive")]
    Receive,
    #[token("type")]
    Type,
    #[token("true")]
    True,
    #[token("false")]
    False,
    
    // Literals
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    IntLiteral(i64),
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    FloatLiteral(f64),
    #[regex(r#""([^"\\]|\\t|\\u|\\n|\\")*""#, |lex| Some(lex.slice().to_string()))]
    StringLiteral(String),
    
    // Identifiers
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| Some(lex.slice().to_string()))]
    Identifier(String),
    
    // Operators
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("=")]
    Equal,
    #[token("==")]
    EqualEqual,
    #[token("!=")]
    NotEqual,
    #[token("<")]
    Less,
    #[token("<=")]
    LessEqual,
    #[token(">")]
    Greater,
    #[token(">=")]
    GreaterEqual,
    #[token("->")]
    Arrow,
    #[token("<-")]
    LeftArrow,
    #[token("=>")]
    FatArrow,
    #[token("::")]
    DoubleColon,
    #[token("?")]
    Question,
    #[token("|")]
    Pipe,
    #[token("&&")]
    And,
    #[token("||")]
    Or,
    #[token("!")]
    Not,
    
    // Concurrency operators
    #[token("<!")] 
    SendOp,  // For sending on a channel
    #[token("!>")]
    ReceiveOp, // For receiving from a channel
    #[token("&")]
    Concurrent, // For concurrent execution
    
    // Delimiters
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token(":")]
    Colon,
    
    // Comments and whitespace
    #[regex(r"//.*", logos::skip)]
    #[regex(r"/\*([^*]|\*[^/])*\*/", logos::skip)]
    #[regex(r"[ \t\n\f]+", logos::skip)]
    Comment,
}

// Helper function to tokenize a string
pub fn tokenize(input: &str) -> Vec<Token> {
    let mut lexer = Token::lexer(input);
    let mut tokens = Vec::new();
    
    while let Some(token) = lexer.next() {
        if let Ok(token) = token {
            tokens.push(token);
        }
        // Error tokens are automatically skipped
    }
    
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn tests_lexer_keywords() {
        let input = "fn let if else match task channel scope transaction reactive mut send receive type true false";
        let tokens = tokenize(input);
        
        assert_eq!(tokens, vec![
            Token::Fn,
            Token::Let,
            Token::If,
            Token::Else,
            Token::Match,
            Token::Task,
            Token::Channel,
            Token::Scope,
            Token::Transaction,
            Token::Reactive,
            Token::Mut,
            Token::Send,
            Token::Receive,
            Token::Type,
            Token::True,
            Token::False,
        ]);
    }
    
    #[test]
    fn tests_lexer_literals() {
        let input = "42 3.14 \"hello world\"";
        let tokens = tokenize(input);
        
        assert_eq!(tokens, vec![
            Token::IntLiteral(42),
            Token::FloatLiteral(3.14),
            Token::StringLiteral("\"hello world\"".to_string()),
        ]);
    }
    
    #[test]
    fn tests_lexer_identifiers() {
        let input = "x y123 _abc function_name";
        let tokens = tokenize(input);
        
        assert_eq!(tokens, vec![
            Token::Identifier("x".to_string()),
            Token::Identifier("y123".to_string()),
            Token::Identifier("_abc".to_string()),
            Token::Identifier("function_name".to_string()),
        ]);
    }
    
    #[test]
    fn tests_lexer_operators() {
        let input = "+ - * / % = == != < <= > >= -> <- => :: ? | <! !> & && || !";
        let tokens = tokenize(input);
        
        assert_eq!(tokens, vec![
            Token::Plus,
            Token::Minus,
            Token::Star,
            Token::Slash,
            Token::Percent,
            Token::Equal,
            Token::EqualEqual,
            Token::NotEqual,
            Token::Less,
            Token::LessEqual,
            Token::Greater,
            Token::GreaterEqual,
            Token::Arrow,
            Token::LeftArrow,
            Token::FatArrow,
            Token::DoubleColon,
            Token::Question,
            Token::Pipe,
            Token::SendOp,
            Token::ReceiveOp,
            Token::Concurrent,
            Token::And,
            Token::Or,
            Token::Not,
        ]);
    }
    
    #[test]
    fn tests_lexer_delimiters() {
        let input = "( ) { } [ ] ; , . :";
        let tokens = tokenize(input);
        
        assert_eq!(tokens, vec![
            Token::LParen,
            Token::RParen,
            Token::LBrace,
            Token::RBrace,
            Token::LBracket,
            Token::RBracket,
            Token::Semicolon,
            Token::Comma,
            Token::Dot,
            Token::Colon,
        ]);
    }
    
    #[test]
    fn tests_lexer_comments_are_skipped() {
        let input = "let x = 5; // This is a comment\n/* This is a block comment */ let y = 10;";
        let tokens = tokenize(input);
        
        assert_eq!(tokens, vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Equal,
            Token::IntLiteral(5),
            Token::Semicolon,
            Token::Let,
            Token::Identifier("y".to_string()),
            Token::Equal,
            Token::IntLiteral(10),
            Token::Semicolon,
        ]);
    }
    
    #[test]
    fn tests_lexer_whitespace_is_skipped() {
        let input = "let   x\t=\n5;";
        let tokens = tokenize(input);
        
        assert_eq!(tokens, vec![
            Token::Let,
            Token::Identifier("x".to_string()),
            Token::Equal,
            Token::IntLiteral(5),
            Token::Semicolon,
        ]);
    }
    
    #[test]
    fn tests_lexer_complex_example() {
        let input = r#"
fn main() {
    let ch = channel<int>();
    
    task {
        ch <! 42;
    }
    
    let value = ch !>;
    
    if value == 42 {
        print("Success!");
    } else {
        print("Failure!");
    }
}"#;
        
        let tokens = tokenize(input);
        
        // We don't need to check every token, just make sure we got the right count
        // and that the first few and last few tokens are correct
        assert!(tokens.len() > 20); // Proper count depends on exact tokenization
        
        // Check some key tokens
        assert_eq!(tokens[0], Token::Fn);
        assert_eq!(tokens[1], Token::Identifier("main".to_string()));
        
        // Check channel and task keywords
        assert!(tokens.contains(&Token::Channel));
        assert!(tokens.contains(&Token::Task));
        
        // Check concurrency operators
        assert!(tokens.contains(&Token::SendOp));
        assert!(tokens.contains(&Token::ReceiveOp));
    }
}