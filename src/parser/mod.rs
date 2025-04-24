use chumsky::prelude::*;
use chumsky::error::Cheap;
use crate::ast::{Expr, Stmt, Program, Type, BinaryOp, UnaryOp, Pattern, MatchCase, FunctionDecl};
use crate::lexer::Token;

/// Create a parser for the entire program
pub fn create_program() -> impl Parser<Token, Program, Error = Cheap<Token>> {
    program_parser()
}

/// Parse literal tokens into their corresponding expression types
fn literal_parser() -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    select! {
        Token::IntLiteral(n) => Expr::IntLiteral(n),
        Token::FloatLiteral(n) => Expr::FloatLiteral(n),
        Token::StringLiteral(s) => Expr::StringLiteral(s),
        Token::True => Expr::BoolLiteral(true),
        Token::False => Expr::BoolLiteral(false),
    }
    .labelled("literal")
}

/// Parse identifiers (variable names)
fn identifier_parser() -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    select! {
        Token::Identifier(name) => Expr::Variable(name),
    }
    .labelled("identifier")
}

/// Parse unary operations: -, !
fn unary_op_parser() -> impl Parser<Token, UnaryOp, Error = Cheap<Token>> + Clone {
    select! {
        Token::Minus => UnaryOp::Negate,
        Token::Not => UnaryOp::Not,
    }
    .labelled("unary operator")
}

/// Parse binary operations: +, -, *, /, %, ==, !=, <, <=, >, >=, &&, ||
fn binary_op_parser() -> impl Parser<Token, BinaryOp, Error = Cheap<Token>> + Clone {
    select! {
        Token::Plus => BinaryOp::Add,
        Token::Minus => BinaryOp::Subtract,
        Token::Star => BinaryOp::Multiply,
        Token::Slash => BinaryOp::Divide,
        Token::Percent => BinaryOp::Modulo,
        Token::EqualEqual => BinaryOp::Equal,
        Token::NotEqual => BinaryOp::NotEqual,
        Token::Less => BinaryOp::Less,
        Token::LessEqual => BinaryOp::LessEqual,
        Token::Greater => BinaryOp::Greater,
        Token::GreaterEqual => BinaryOp::GreaterEqual,
        Token::And => BinaryOp::And,
        Token::Or => BinaryOp::Or,
    }
    .labelled("binary operator")
}

/// Parse type annotations
fn type_parser() -> impl Parser<Token, Type, Error = Cheap<Token>> + Clone {
    recursive(|type_parser| {
        // Named types (like "int", "float", "string", "bool", or custom types)
        let named = select! {
            Token::Identifier(name) => Type::Named(name),
        };
        
        // Generic types (like "List<T>")
        let generic = select! {
            Token::Identifier(name) => name,
        }
        .then_ignore(just(Token::Less))
        .then(type_parser.clone().separated_by(just(Token::Comma)).collect::<Vec<_>>())
        .then_ignore(just(Token::Greater))
        .map(|(name, args)| Type::Generic(name, args));
        
        // Function types (like "fn(A, B) -> C")
        let function = just(Token::Fn)
            .ignore_then(
                just(Token::LParen)
                    .ignore_then(type_parser.clone().separated_by(just(Token::Comma)).collect::<Vec<_>>())
                    .then_ignore(just(Token::RParen))
            )
            .then_ignore(just(Token::Arrow))
            .then(type_parser.clone())
            .map(|(args, ret)| Type::Function(args, Box::new(ret)));
        
        choice((generic, function, named)).labelled("type")
    })
}

/// Parse patterns for match expressions
fn pattern_parser() -> impl Parser<Token, Pattern, Error = Cheap<Token>> + Clone {
    recursive(|pattern_parser| {
        // Wildcard pattern: _
        let wildcard = just(Token::Identifier("_".to_string()))
            .map(|_| Pattern::Wildcard);
        
        // Variable pattern: x
        let variable = select! {
            Token::Identifier(name) if name != "_" => Pattern::Variable(name),
        };
        
        // Literal pattern
        let literal = literal_parser().map(Pattern::Literal);
        
        // Constructor pattern: Some(x)
        let constructor = select! {
            Token::Identifier(name) => name,
        }
        .then(
            just(Token::LParen)
                .ignore_then(pattern_parser.clone())
                .then_ignore(just(Token::RParen))
                .or_not()
        )
        .map(|(name, arg)| Pattern::Constructor(name, arg.map(Box::new)));
        
        choice((wildcard, constructor, variable, literal)).labelled("pattern")
    })
}

/// Main expression parser with precedence handling
fn expr_parser() -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    recursive(|expr| {
        // First, handle the atomic expressions
        let atom = atom_parser(expr.clone());
        
        // Handle function calls
        let func_call = function_call_parser(atom.clone());
        
        // Handle unary operations
        let unary = unary_expr_parser(func_call.clone());
        
        // Then build the binary operations with precedence
        let mul_div = multiplicative_expr_parser(unary.clone());
        let add_sub = additive_expr_parser(mul_div.clone());
        let comparison = comparison_expr_parser(add_sub.clone());
        let and_expr = and_expr_parser(comparison.clone());
        let or_expr = or_expr_parser(and_expr.clone());
        
        or_expr
    })
}

/// Parse atomic expressions (literals, variables, blocks, etc.)
fn atom_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone + 'static,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    let literal = literal_parser();
    let identifier = identifier_parser();
    
    // Parenthesized expressions
    let paren_expr = just(Token::LParen)
        .ignore_then(expr.clone())
        .then_ignore(just(Token::RParen));
    
    // Block expressions
    let block = block_parser(expr.clone());
    
    // Control flow expressions
    let if_expr = if_expr_parser(expr.clone());
    let match_expr = match_expr_parser(expr.clone());
    
    // Concurrency primitives
    let task_expr = task_expr_parser(expr.clone());
    let channel_expr = channel_expr_parser();
    let send_expr = send_expr_parser(expr.clone());
    let receive_expr = receive_expr_parser(expr.clone());
    let scope_expr = scope_expr_parser(expr.clone());
    let transaction_expr = transaction_expr_parser(expr.clone());
    let reactive_expr = reactive_expr_parser(expr.clone());
    
    choice((
        literal,
        identifier,
        paren_expr,
        block,
        if_expr,
        match_expr,
        task_expr,
        channel_expr,
        send_expr,
        receive_expr,
        scope_expr,
        transaction_expr,
        reactive_expr,
    ))
    .labelled("expression")
}

/// Parse unary expressions (-, !)
fn unary_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    unary_op_parser()
        .then(expr.clone())
        .map(|(op, expr)| Expr::Unary(op, Box::new(expr)))
        .or(expr)
}

/// Parse function calls
fn function_call_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    expr.clone()
        .then(
            just(Token::LParen)
                .ignore_then(expr.separated_by(just(Token::Comma)).collect::<Vec<_>>())
                .then_ignore(just(Token::RParen))
                .or_not()
        )
        .map(|(func, args_opt)| {
            if let Some(args) = args_opt {
                Expr::FunctionCall {
                    function: Box::new(func),
                    arguments: args,
                }
            } else {
                func
            }
        })
}

/// Parse block expressions { ... }
fn block_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone + 'static,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    just(Token::LBrace)
        .ignore_then(
            stmt_parser(expr.clone())
                .repeated()
                .collect::<Vec<Stmt>>()
                .then(expr.or_not())
        )
        .then_ignore(just(Token::RBrace))
        .map(|(statements, result)| Expr::Block {
            statements,
            result: result.map(Box::new),
        })
        .labelled("block")
}

/// Parse if expressions
fn if_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone + 'static,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    recursive(|if_expr| {
        just(Token::If)
            .ignore_then(expr.clone())
            .then(block_parser(expr.clone()))
            .then(
                just(Token::Else)
                    .ignore_then(
                        block_parser(expr.clone())
                        .or(if_expr)
                    )
                    .or_not()
            )
            .map(|((condition, then_branch), else_branch)| {
                Expr::If {
                    condition: Box::new(condition),
                    then_branch: Box::new(then_branch),
                    else_branch: else_branch.map(Box::new),
                }
            })
    })
}

/// Parse match expressions
fn match_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone + 'static,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    just(Token::Match)
        .ignore_then(expr.clone())
        .then_ignore(just(Token::LBrace))
        .then(
            pattern_parser()
                .then_ignore(just(Token::FatArrow))
                .then(expr.clone())
                .then_ignore(just(Token::Comma).or_not())
                .repeated()
                .collect::<Vec<_>>()
        )
        .then_ignore(just(Token::RBrace))
        .map(|(value, cases)| {
            let match_cases = cases.into_iter().map(|(pattern, body)| {
                MatchCase { pattern, body }
            }).collect();
            
            Expr::Match {
                value: Box::new(value),
                cases: match_cases,
            }
        })
}

/// Parse task expressions
fn task_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone + 'static,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    just(Token::Task)
        .ignore_then(block_parser(expr))
        .map(|body| Expr::Task {
            body: Box::new(body)
        })
}

/// Parse channel expressions
fn channel_expr_parser() -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    just(Token::Channel)
        .ignore_then(just(Token::Less))
        .ignore_then(type_parser())
        .then_ignore(just(Token::Greater))
        .then(
            just(Token::LParen)
                .ignore_then(expr_parser())
                .then_ignore(just(Token::RParen))
                .or_not()
        )
        .map(|(typ, capacity)| Expr::Channel {
            typ,
            capacity: capacity.map(Box::new),
        })
}

/// Parse send expressions
/// Assuming the syntax is "send value -> channel" instead of "send value to channel"
fn send_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    just(Token::Send)
        .ignore_then(expr.clone())
        .then_ignore(just(Token::Arrow)) // Using Arrow token instead of To
        .then(expr)
        .map(|(value, channel)| {
            Expr::Send {
                value: Box::new(value),
                channel: Box::new(channel),
            }
        })
}

/// Parse receive expressions
fn receive_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    just(Token::Receive)
        .ignore_then(expr)
        .map(|channel| Expr::Receive {
            channel: Box::new(channel)
        })
}

/// Parse scope expressions
fn scope_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone + 'static,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    just(Token::Scope)
        .ignore_then(block_parser(expr))
        .map(|body| Expr::Scope {
            body: Box::new(body)
        })
}

/// Parse transaction expressions
fn transaction_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone + 'static,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    just(Token::Transaction)
        .ignore_then(block_parser(expr))
        .map(|body| Expr::Transaction {
            body: Box::new(body)
        })
}

/// Parse reactive expressions
fn reactive_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    just(Token::Reactive)
        .ignore_then(expr)
        .map(|value| Expr::Reactive {
            value: Box::new(value)
        })
}

/// Parse multiplicative expressions (*, /, %)
fn multiplicative_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    let op = just(Token::Star).to(BinaryOp::Multiply)
        .or(just(Token::Slash).to(BinaryOp::Divide))
        .or(just(Token::Percent).to(BinaryOp::Modulo));
    
    binary_expr_parser(expr, op)
}

/// Parse additive expressions (+, -)
fn additive_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    let op = just(Token::Plus).to(BinaryOp::Add)
        .or(just(Token::Minus).to(BinaryOp::Subtract));
    
    binary_expr_parser(expr, op)
}

/// Parse comparison expressions (<, <=, >, >=, ==, !=)
fn comparison_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    let op = just(Token::Less).to(BinaryOp::Less)
        .or(just(Token::LessEqual).to(BinaryOp::LessEqual))
        .or(just(Token::Greater).to(BinaryOp::Greater))
        .or(just(Token::GreaterEqual).to(BinaryOp::GreaterEqual))
        .or(just(Token::EqualEqual).to(BinaryOp::Equal))
        .or(just(Token::NotEqual).to(BinaryOp::NotEqual));
    
    binary_expr_parser(expr, op)
}

/// Parse logical AND expressions
fn and_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    let op = just(Token::And).to(BinaryOp::And);
    
    binary_expr_parser(expr, op)
}

/// Parse logical OR expressions
fn or_expr_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    let op = just(Token::Or).to(BinaryOp::Or);
    
    binary_expr_parser(expr, op)
}

/// Helper for binary expression parsing with left associativity
fn binary_expr_parser(
    atom: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone,
    op: impl Parser<Token, BinaryOp, Error = Cheap<Token>> + Clone,
) -> impl Parser<Token, Expr, Error = Cheap<Token>> + Clone {
    atom.clone()
        .then(op.then(atom).repeated())
        .map(|(first, rest)| {
            rest.into_iter().fold(first, |lhs, (op, rhs)| {
                Expr::Binary(Box::new(lhs), op, Box::new(rhs))
            })
        })
}

/// Parse statements (let, function declaration, type declaration, expression statements)
fn stmt_parser(
    expr: impl Parser<Token, Expr, Error = Cheap<Token>> + Clone + 'static,
) -> impl Parser<Token, Stmt, Error = Cheap<Token>> + Clone {
    // Let statement
    let let_stmt = just(Token::Let)
        .ignore_then(select! {
            Token::Identifier(name) => name,
        })
        .then_ignore(just(Token::Equal))
        .then(expr.clone())
        .then_ignore(just(Token::Semicolon))
        .map(|(name, value)| Stmt::Let(name, value));
    
    // Function declaration
    let fn_decl = recursive(|fn_decl| {
        // Remove this variable entirely to avoid the complex type annotation
        // let stmt_choices = choice((
        //     let_stmt.clone(),
        //     fn_decl,
        //     expr.clone().then_ignore(just(Token::Semicolon)).map(Stmt::Expr),
        // ));
        
        just(Token::Fn)
            .ignore_then(select! {
                Token::Identifier(name) => name,
            })
            .then_ignore(just(Token::LParen))
            .then(
                select! {
                    Token::Identifier(name) => name,
                }
                .then_ignore(just(Token::Colon))
                .then(type_parser())
                .map(|(name, typ)| (name, typ))
                .separated_by(just(Token::Comma))
                .collect::<Vec<_>>()
            )
            .then_ignore(just(Token::RParen))
            .then(
                just(Token::Arrow)
                    .ignore_then(type_parser())
                    .or_not()
            )
            .then(block_parser(expr.clone()))
            .map(|(((name, parameters), return_type), body)| {
                Stmt::FunctionDecl(FunctionDecl {
                    name,
                    parameters,
                    return_type,
                    body,
                })
            })
    });
    
    // Type declaration
    let type_decl = just(Token::Type)
        .ignore_then(select! {
            Token::Identifier(name) => name,
        })
        .then(
            just(Token::Less)
                .ignore_then(
                    select! {
                        Token::Identifier(name) => name,
                    }
                    .separated_by(just(Token::Comma))
                    .collect::<Vec<_>>()
                )
                .then_ignore(just(Token::Greater))
                .or_not()
                .map(|opt| opt.unwrap_or_else(Vec::new))
        )
        .then_ignore(just(Token::Equal))
        .then(type_parser())
        .then_ignore(just(Token::Semicolon))
        .map(|((name, type_params), typ)| {
            Stmt::TypeDecl(name, type_params, typ)
        });
    
    // Expression statement
    let expr_stmt = expr
        .then_ignore(just(Token::Semicolon))
        .map(Stmt::Expr);
    
    choice((let_stmt, fn_decl, type_decl, expr_stmt))
        .labelled("statement")
}

/// Parse complete programs
fn program_parser() -> impl Parser<Token, Program, Error = Cheap<Token>> {
    stmt_parser(expr_parser())
        .repeated()
        .collect::<Vec<_>>()
        .map(|statements| Program { statements })
        .labelled("program")
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::{Token, tokenize};
    use chumsky::Parser;

   // Helper function to parse expressions from a string
fn parse_expr(input: &str) -> Result<Expr, Vec<Cheap<Token>>> {
    let tokens = tokenize(input);
    expr_parser().parse(tokens)
}

// Helper function to parse statements from a string
fn parse_stmt(input: &str) -> Result<Stmt, Vec<Cheap<Token>>> {
    let tokens = tokenize(input);
    stmt_parser(expr_parser()).parse(tokens)
}

// Helper function to parse a complete program from a string
fn parse_program(input: &str) -> Result<Program, Vec<Cheap<Token>>> {
    let tokens = tokenize(input);
    program_parser().parse(tokens)
}

    #[test]
    fn parser_test_literals() {
        assert!(matches!(parse_expr("42"), Ok(Expr::IntLiteral(42))));
        assert!(matches!(parse_expr("3.14"), Ok(Expr::FloatLiteral(3.14))));
        assert!(matches!(parse_expr("\"hello\""), Ok(Expr::StringLiteral(s)) if s == "hello"));
        assert!(matches!(parse_expr("true"), Ok(Expr::BoolLiteral(true))));
        assert!(matches!(parse_expr("false"), Ok(Expr::BoolLiteral(false))));
    }

    #[test]
    fn parser_test_variables() {
        assert!(matches!(parse_expr("x"), Ok(Expr::Variable(s)) if s == "x"));
        assert!(matches!(parse_expr("someVar"), Ok(Expr::Variable(s)) if s == "someVar"));
    }

    #[test]
    fn parser_test_binary_ops() {
        // Test addition
        if let Ok(Expr::Binary(left, op, right)) = parse_expr("1 + 2") {
            assert!(matches!(*left, Expr::IntLiteral(1)));
            assert!(matches!(op, BinaryOp::Add));
            assert!(matches!(*right, Expr::IntLiteral(2)));
        } else {
            panic!("Failed to parse binary expression");
        }

        // Test precedence
        if let Ok(Expr::Binary(left, op, right)) = parse_expr("1 + 2 * 3") {
            assert!(matches!(*left, Expr::IntLiteral(1)));
            assert!(matches!(op, BinaryOp::Add));
            
            if let Expr::Binary(left_inner, op_inner, right_inner) = *right {
                assert!(matches!(*left_inner, Expr::IntLiteral(2)));
                assert!(matches!(op_inner, BinaryOp::Multiply));
                assert!(matches!(*right_inner, Expr::IntLiteral(3)));
            } else {
                panic!("Failed to parse nested binary expression");
            }
        } else {
            panic!("Failed to parse complex binary expression");
        }
    }

    #[test]
    fn parser_test_unary_ops() {
        // Test negation
        if let Ok(Expr::Unary(op, expr)) = parse_expr("-42") {
            assert!(matches!(op, UnaryOp::Negate));
            assert!(matches!(*expr, Expr::IntLiteral(42)));
        } else {
            panic!("Failed to parse unary expression");
        }

        // Test logical not
        if let Ok(Expr::Unary(op, expr)) = parse_expr("!true") {
            assert!(matches!(op, UnaryOp::Not));
            assert!(matches!(*expr, Expr::BoolLiteral(true)));
        } else {
            panic!("Failed to parse unary not expression");
        }
    }

    #[test]
    fn parser_test_function_call() {
        // Test simple function call
        if let Ok(Expr::FunctionCall { function, arguments }) = parse_expr("foo()") {
            assert!(matches!(*function, Expr::Variable(s) if s == "foo"));
            assert_eq!(arguments.len(), 0);
        } else {
            panic!("Failed to parse function call");
        }

        // Test function call with arguments
        if let Ok(Expr::FunctionCall { function, arguments }) = parse_expr("bar(1, 2, 3)") {
            assert!(matches!(*function, Expr::Variable(s) if s == "bar"));
            assert_eq!(arguments.len(), 3);
            assert!(matches!(arguments[0], Expr::IntLiteral(1)));
            assert!(matches!(arguments[1], Expr::IntLiteral(2)));
            assert!(matches!(arguments[2], Expr::IntLiteral(3)));
        } else {
            panic!("Failed to parse function call with arguments");
        }
    }

    #[test]
    fn parser_test_block() {
        // Test empty block
        if let Ok(Expr::Block { statements, result }) = parse_expr("{}") {
            assert_eq!(statements.len(), 0);
            assert!(result.is_none());
        } else {
            panic!("Failed to parse empty block");
        }

        // Test block with statements and expression
        if let Ok(Expr::Block { statements, result }) = parse_expr("{ x; y; 42 }") {
            assert_eq!(statements.len(), 2);
            assert!(result.is_some());
            if let Some(expr) = result {
                assert!(matches!(*expr, Expr::IntLiteral(42)));
            }
        } else {
            panic!("Failed to parse block with statements and expression");
        }
    }

    #[test]
    fn parser_test_if_expr() {
        // Test if without else
        let code = "if true { 42 }";
        if let Ok(Expr::If { condition, then_branch, else_branch }) = parse_expr(code) {
            assert!(matches!(*condition, Expr::BoolLiteral(true)));
            if let Expr::Block { statements, result } = *then_branch {
                assert_eq!(statements.len(), 0);
                assert!(result.is_some());
                if let Some(expr) = result {
                    assert!(matches!(*expr, Expr::IntLiteral(42)));
                }
            } else {
                panic!("Then branch is not a block");
            }
            assert!(else_branch.is_none());
        } else {
            panic!("Failed to parse if expression");
        }

        // Test if with else
        let code = "if x { 1 } else { 2 }";
        if let Ok(Expr::If { condition, then_branch, else_branch }) = parse_expr(code) {
            assert!(matches!(*condition, Expr::Variable(s) if s == "x"));
            assert!(else_branch.is_some());
        } else {
            panic!("Failed to parse if-else expression");
        }
    }

    #[test]
    fn parser_test_let_statement() {
        // Test let statement
        let code = "let x = 42;";
        if let Ok(Stmt::Let(name, value)) = parse_stmt(code) {
            assert_eq!(name, "x");
            assert!(matches!(value, Expr::IntLiteral(42)));
        } else {
            panic!("Failed to parse let statement");
        }
    }

    #[test]
    fn parser_test_expr_statement() {
        // Test expression statement
        let code = "foo();";
        if let Ok(Stmt::Expr(Expr::FunctionCall { function, arguments })) = parse_stmt(code) {
            assert!(matches!(*function, Expr::Variable(s) if s == "foo"));
            assert_eq!(arguments.len(), 0);
        } else {
            panic!("Failed to parse expression statement");
        }
    }

    #[test]
    fn parser_test_simple_program() {
        // Test a simple program
        let code = "let x = 10; let y = 20; x + y;";
        if let Ok(Program { statements }) = parse_program(code) {
            assert_eq!(statements.len(), 3);
            assert!(matches!(statements[0], Stmt::Let(..)));
            assert!(matches!(statements[1], Stmt::Let(..)));
            assert!(matches!(statements[2], Stmt::Expr(..)));
        } else {
            panic!("Failed to parse simple program");
        }
    }

    #[test]
    fn parser_test_task() {
        // Test task expression
        let code = "task { 42 }";
        if let Ok(Expr::Task { body }) = parse_expr(code) {
            if let Expr::Block { statements, result } = *body {
                assert_eq!(statements.len(), 0);
                assert!(result.is_some());
                if let Some(expr) = result {
                    assert!(matches!(*expr, Expr::IntLiteral(42)));
                }
            } else {
                panic!("Task body is not a block");
            }
        } else {
            panic!("Failed to parse task expression");
        }
    }

    #[test]
    fn parser_test_channel() {
        // Test channel expression
        let code = "channel<int>";
        if let Ok(Expr::Channel { typ, capacity }) = parse_expr(code) {
            assert!(matches!(typ, Type::Named(s) if s == "int"));
            assert!(capacity.is_none());
        } else {
            panic!("Failed to parse channel expression");
        }
    }

    #[test]
    fn parser_test_send_receive() {
        // Test send expression
        let code = "send 42 -> ch";
        if let Ok(Expr::Send { value, channel }) = parse_expr(code) {
            assert!(matches!(*value, Expr::IntLiteral(42)));
            assert!(matches!(*channel, Expr::Variable(s) if s == "ch"));
        } else {
            panic!("Failed to parse send expression");
        }

        // Test receive expression
        let code = "receive ch";
        if let Ok(Expr::Receive { channel }) = parse_expr(code) {
            assert!(matches!(*channel, Expr::Variable(s) if s == "ch"));
        } else {
            panic!("Failed to parse receive expression");
        }
    }
}