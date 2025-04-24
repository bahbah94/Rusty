// src/ast/mod.rs
#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    
    // Variables and operations
    Variable(String),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Unary(UnaryOp, Box<Expr>),
    
    // Function-related
    FunctionCall {
        function: Box<Expr>,
        arguments: Vec<Expr>,
    },
    
    // Control flow
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },
    
    // Concurrency primitives
    Task {
        body: Box<Expr>,
    },
    Channel {
        typ: Type,
        capacity: Option<Box<Expr>>,
    },
    Send {
        channel: Box<Expr>,
        value: Box<Expr>,
    },
    Receive {
        channel: Box<Expr>,
    },
    Scope {
        body: Box<Expr>,
    },
    Transaction {
        body: Box<Expr>,
    },
    Reactive {
        value: Box<Expr>,
    },
    
    // Pattern matching
    Match {
        value: Box<Expr>,
        cases: Vec<MatchCase>,
    },
    
    // Blocks containing multiple expressions
    Block {
        statements: Vec<Stmt>,
        result: Option<Box<Expr>>,
    },
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(String, Expr),
    FunctionDecl(FunctionDecl),
    TypeDecl(String, Vec<String>, Type),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub name: String,
    pub parameters: Vec<(String, Type)>,
    pub return_type: Option<Type>,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add, Subtract, Multiply, Divide, Modulo,
    Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual,
    And, Or,
}

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Negate, Not,
}

#[derive(Debug, Clone)]
pub enum Type {
    Named(String),
    Generic(String, Vec<Type>),
    Function(Vec<Type>, Box<Type>),
}

#[derive(Debug, Clone)]
pub struct MatchCase {
    pub pattern: Pattern,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Literal(Expr),
    Variable(String),
    Constructor(String, Option<Box<Pattern>>),
}

#[derive(Debug, Clone)]
pub struct Program {
    pub statements: Vec<Stmt>,
}

impl Program {
    pub fn new(statements: Vec<Stmt>) -> Self {
        Self { statements }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_construction() {
        // Test creating basic expressions
        let literal = Expr::IntLiteral(42);
        let variable = Expr::Variable("x".to_string());
        
        // Test binary operation with boxed expressions
        let binary = Expr::Binary(
            Box::new(literal), 
            BinaryOp::Add, 
            Box::new(variable)
        );
        
        // Test more complex nesting
        let if_expr = Expr::If {
            condition: Box::new(binary),
            then_branch: Box::new(Expr::StringLiteral("then".to_string())),
            else_branch: Some(Box::new(Expr::StringLiteral("else".to_string()))),
        };
        
        // Verify structure is correct (very basic)
        if let Expr::If { condition, .. } = if_expr {
            if let Expr::Binary(left, op, _) = *condition {
                if let Expr::IntLiteral(value) = *left {
                    assert_eq!(value, 42);
                } else {
                    panic!("Expected IntLiteral");
                }
                
                match op {
                    BinaryOp::Add => (),
                    _ => panic!("Expected Add operator"),
                }
            } else {
                panic!("Expected Binary expression");
            }
        } else {
            panic!("Expected If expression");
        }
    }
}